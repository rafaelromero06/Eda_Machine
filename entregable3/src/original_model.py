import numbers

import numpy as np
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def nucleo_umbrales(umbrales, centro, ancho, escala):
    """Pesos gaussianos normalizados sobre los umbrales ordinales.

    Parameters
    ----------
    umbrales : array de forma (K,)
        Valores t_k de los indicadores acumulados 1{y >= t_k}.
    centro : float
        Centro del núcleo, en las unidades de y.
    ancho : float
        Ancho relativo h >= 0. ``0`` da todo el peso al umbral más cercano al
        centro y ``np.inf`` reparte el peso por igual.
    escala : float
        Desviación estándar de y; el ancho efectivo es ``ancho * escala``.

    Returns
    -------
    ndarray de forma (K,) con pesos no negativos que suman 1.
    """
    t = np.asarray(umbrales, dtype=float)
    if np.isinf(ancho):
        w = np.ones_like(t)
    elif ancho == 0:
        w = np.zeros_like(t)
        w[np.argmin(np.abs(t - centro))] = 1.0
    else:
        z = -0.5 * ((t - centro) / (ancho * escala)) ** 2
        w = np.exp(z - z.max())
    return w / w.sum()


def auc_mann_whitney(y, puntaje):
    """AUC-ROC como estadístico de Mann-Whitney con rangos promedio en empates."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(puntaje)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ajustar_platt(z, y, max_iter=100, tol=1e-12):
    """Regresión logística de una variable, p = sigma(a + b z), por Newton con búsqueda lineal.

    Returns
    -------
    a, b : float
        Parámetros que minimizan la log-pérdida (problema convexo).
    historia : list
        Log-pérdida tras cada iteración; nunca aumenta por la condición de Armijo.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    D = np.column_stack([np.ones_like(z), z])

    def perdida(th):
        eta = D @ th
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    th = np.array([0.0, 1.0])
    f = perdida(th)
    historia = [f]
    for _ in range(max_iter):
        p = expit(D @ th)
        g = D.T @ (p - y)
        H = (D * (p * (1 - p))[:, None]).T @ D + 1e-10 * np.eye(2)
        paso = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-10:
            nuevo = th - t * paso
            f_nuevo = perdida(nuevo)
            if f_nuevo <= f - 1e-4 * t * float(g @ paso):
                break
            t *= 0.5
        else:
            break
        mejora = f - f_nuevo
        th, f = nuevo, f_nuevo
        historia.append(f)
        if mejora <= tol * (1.0 + abs(f)):
            break
    return float(th[0]), float(th[1]), historia


class BosqueOrdinalFocalizado(ClassifierMixin, BaseEstimator):
    """Bosque Ordinal Focalizado (BOF) para un evento binario definido por un umbral ordinal.

    El objetivo es el evento Y = 1{y >= umbral}, donde y es una respuesta
    ordinal (por ejemplo, la nota de calidad de 3 a 9). En lugar de entrenar
    con Y, el bosque aprende todos los indicadores acumulados
    Z_k = 1{y >= t_k} con árboles extremadamente aleatorizados de salida
    múltiple. Tres piezas forman el método:

    1. Criterio de partición: impureza acumulada ponderada
       G(S) = sum_k w_k F_k(S) (1 - F_k(S)), con pesos w de un núcleo gaussiano
       centrado en el umbral del evento (``ancho_particion``).
    2. Agregación focalizada: el puntaje es s(x) = sum_k omega_k P_k(x), una
       combinación de las probabilidades acumuladas del bosque con pesos de
       un segundo núcleo (``ancho_agregacion``) que toma información de los
       umbrales vecinos al del evento.
    3. Calibración fuera de bolsa: p(x) = sigma(a + b logit s(x)), con (a, b)
       ajustados por máxima verosimilitud sobre los puntajes fuera de bolsa.

    Parameters
    ----------
    umbral : float o None
        Nivel que define el evento (y >= umbral). Con y binaria (0/1) puede
        omitirse y vale 1.
    n_estimators : int
        Número de árboles.
    max_features : {"sqrt", "log2"}, int, float o None
        Atributos candidatos por nodo.
    n_cortes : int
        Cortes aleatorios por atributo candidato (1 = Extra-Trees clásico).
    min_samples_leaf : int
        Mínimo de muestras distintas en cada hoja.
    max_depth : int o None
        Profundidad máxima.
    max_samples : float o None
        Fracción del tamaño de la muestra bootstrap (None = 1).
    ancho_particion, ancho_agregacion : float
        Anchos relativos de los núcleos (en desviaciones estándar de y);
        ``0`` usa solo el umbral del evento y ``np.inf`` pesa todos por igual.
    desplazamiento : float
        Corrimiento del centro de ambos núcleos, en desviaciones estándar de y.
    calibrar : bool
        Si es True, aplica la calibración fuera de bolsa.
    max_umbrales : int
        Máximo de indicadores acumulados; con más niveles se eligen por cuantiles.
    tamano_lote : int
        Árboles que se construyen a la vez (solo afecta memoria y velocidad).
    random_state : int, Generator o None
        Semilla.
    """

    def __init__(self, umbral=None, n_estimators=300, max_features="sqrt", n_cortes=1,
                 min_samples_leaf=1, max_depth=None, max_samples=None, ancho_particion=np.inf,
                 ancho_agregacion=0.7, desplazamiento=0.0, calibrar=True, max_umbrales=24,
                 tamano_lote=50, random_state=None):
        self.umbral = umbral
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_cortes = n_cortes
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.ancho_particion = ancho_particion
        self.ancho_agregacion = ancho_agregacion
        self.desplazamiento = desplazamiento
        self.calibrar = calibrar
        self.max_umbrales = max_umbrales
        self.tamano_lote = tamano_lote
        self.random_state = random_state

    def _validar_hiperparametros(self, p):
        def entero(nombre, minimo):
            v = getattr(self, nombre)
            if not isinstance(v, numbers.Integral) or isinstance(v, bool) or v < minimo:
                raise ValueError(f"{nombre} debe ser un entero >= {minimo}; se recibió {v!r}")

        entero("n_estimators", 1)
        entero("n_cortes", 1)
        entero("min_samples_leaf", 1)
        entero("max_umbrales", 1)
        entero("tamano_lote", 1)
        if self.max_depth is not None:
            entero("max_depth", 1)
        for nombre in ("ancho_particion", "ancho_agregacion"):
            v = getattr(self, nombre)
            if not isinstance(v, numbers.Real) or np.isnan(v) or v < 0:
                raise ValueError(f"{nombre} debe ser un número >= 0 (np.inf permitido); se recibió {v!r}")
        if not isinstance(self.desplazamiento, numbers.Real) or not np.isfinite(self.desplazamiento):
            raise ValueError(f"desplazamiento debe ser un número finito; se recibió {self.desplazamiento!r}")
        if self.max_samples is not None and not (isinstance(self.max_samples, numbers.Real)
                                                 and 0 < self.max_samples <= 1):
            raise ValueError(f"max_samples debe estar en (0, 1]; se recibió {self.max_samples!r}")
        mf = self.max_features
        if mf is None:
            m = p
        elif mf == "sqrt":
            m = int(np.sqrt(p))
        elif mf == "log2":
            m = int(np.log2(p))
        elif isinstance(mf, numbers.Integral) and not isinstance(mf, bool):
            if mf < 1:
                raise ValueError(f"max_features entero debe ser >= 1; se recibió {mf!r}")
            m = mf
        elif isinstance(mf, numbers.Real) and 0 < mf <= 1:
            m = int(round(mf * p))
        else:
            raise ValueError(f"max_features no válido: {mf!r}")
        return int(min(p, max(1, m)))

    def _definir_umbrales(self, y):
        niveles = np.unique(y)
        if niveles.size < 2:
            raise ValueError("y debe tener al menos dos niveles distintos")
        if self.umbral is None:
            if not np.isin(niveles, [0, 1]).all():
                raise ValueError("umbral es obligatorio cuando y no es binaria 0/1")
            tau = 1.0
        else:
            tau = float(self.umbral)
        if not (niveles[0] < tau <= niveles[-1]):
            raise ValueError(f"umbral={tau} debe estar entre el mínimo (exclusivo) y el máximo de y")
        candidatos = niveles[1:].astype(float)
        if candidatos.size > self.max_umbrales:
            q = np.quantile(y, np.linspace(0, 1, self.max_umbrales + 2)[1:-1], method="higher")
            candidatos = np.unique(q[q > niveles[0]])
        umbrales = np.unique(np.append(candidatos, tau))
        return umbrales, tau, int(np.flatnonzero(umbrales == tau)[0])

    def fit(self, X, y):
        """Ajusta el bosque con la respuesta ordinal y (o binaria 0/1).

        Returns
        -------
        self
        """
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_all_finite=True, y_numeric=True)
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("y contiene valores no finitos")
        n, p = X.shape
        self._m = self._validar_hiperparametros(p)
        self.umbrales_, self.umbral_evento_, self.indice_evento_ = self._definir_umbrales(y)
        evento = (y >= self.umbral_evento_).astype(int)
        if evento.min() == evento.max():
            raise ValueError("los datos de entrenamiento deben contener las dos clases del evento")
        self.classes_ = np.array([0, 1])
        self.escala_ = float(np.std(y)) if np.std(y) > 0 else 1.0
        self.centro_ = self.umbral_evento_ + self.desplazamiento * self.escala_
        self.pesos_particion_ = nucleo_umbrales(self.umbrales_, self.centro_, self.ancho_particion, self.escala_)
        self.pesos_agregacion_ = nucleo_umbrales(self.umbrales_, self.centro_, self.ancho_agregacion, self.escala_)
        Z = (y[:, None] >= self.umbrales_[None, :]).astype(float)

        rng = np.random.default_rng(self.random_state)
        tam = n if self.max_samples is None else max(1, int(round(self.max_samples * n)))
        partes, conteos = [], []
        importancias = np.zeros(p)
        hechos = 0
        while hechos < self.n_estimators:
            b = min(self.tamano_lote, self.n_estimators - hechos)
            c = np.stack([np.bincount(rng.integers(0, n, tam), minlength=n) for _ in range(b)])
            partes.append(self._crecer_lote(X, Z, c, rng, importancias))
            conteos.append(c)
            hechos += b
        self._ensamblar(partes)
        self.feature_importances_ = importancias / importancias.sum() if importancias.sum() > 0 else importancias
        self._fuera_de_bolsa(X, evento, np.concatenate(conteos))
        return self

    def _crecer_lote(self, X, Z, conteos, rng, importancias):
        n, p = X.shape
        K = Z.shape[1]
        m, r = self._m, self.n_cortes
        C = m * r
        w_s = self.pesos_particion_
        hoja_min = self.min_samples_leaf
        arbol_fila, fila = np.nonzero(conteos)
        peso = conteos[arbol_fila, fila].astype(float)
        j = arbol_fila.astype(np.int64)
        J = conteos.shape[0]
        niveles = []
        profundidad = 0
        while J > 0:
            filas_nodo = np.bincount(j, minlength=J).astype(float)
            w_nodo = np.bincount(j, weights=peso, minlength=J)
            A = sparse.csr_matrix((np.ones(j.size), (j, np.arange(j.size))), shape=(J, j.size))
            S = A @ (peso[:, None] * Z[fila])
            F = S / w_nodo[:, None]
            G = (F * (1.0 - F)) @ w_s
            divisible = (filas_nodo >= 2 * hoja_min) & (G > 1e-12)
            if self.max_depth is not None and profundidad >= self.max_depth:
                divisible[:] = False
            atributo = np.full(J, -1, dtype=np.int64)
            corte = np.full(J, np.nan)
            rango = np.full(J, -1, dtype=np.int64)
            nodos_d = np.flatnonzero(divisible)
            if nodos_d.size:
                Jd = nodos_d.size
                pos = np.full(J, -1, dtype=np.int64)
                pos[nodos_d] = np.arange(Jd)
                sel = pos[j] >= 0
                jr, fr, pr = pos[j[sel]], fila[sel], peso[sel]
                orden = np.argsort(jr, kind="stable")
                inicio = np.searchsorted(jr[orden], np.arange(Jd))
                Xo = X[fr[orden]]
                mins = np.minimum.reduceat(Xo, inicio, axis=0)
                maxs = np.maximum.reduceat(Xo, inicio, axis=0)
                varia = maxs > mins
                clave = rng.random((Jd, p)) + 2.0 * (~varia)
                cand = np.argsort(clave, axis=1)[:, :m]
                filas_d = np.arange(Jd)[:, None]
                lo, hi = mins[filas_d, cand], maxs[filas_d, cand]
                cortes = lo[:, :, None] + rng.random((Jd, m, r)) * (hi - lo)[:, :, None]
                V = X[fr[:, None], cand[jr]]
                izq = (V[:, :, None] < cortes[jr]).reshape(jr.size, C)
                fi, ci = np.nonzero(izq)
                Q = sparse.csr_matrix((np.ones(fi.size), (jr[fi] * C + ci, fi)), shape=(Jd * C, jr.size))
                E = Q @ np.column_stack([np.ones(jr.size), pr, pr[:, None] * Z[fr]])
                nL, wL = E[:, 0].reshape(Jd, C), E[:, 1].reshape(Jd, C)
                SL = E[:, 2:].reshape(Jd, C, K)
                nR = filas_nodo[nodos_d][:, None] - nL
                wR = w_nodo[nodos_d][:, None] - wL
                SR = S[nodos_d][:, None, :] - SL
                valido = (nL >= hoja_min) & (nR >= hoja_min) & np.repeat(hi > lo, r, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ganancia = ((SL ** 2 / wL[:, :, None] + SR ** 2 / wR[:, :, None]) @ w_s
                                - ((S[nodos_d] ** 2 / w_nodo[nodos_d][:, None]) @ w_s)[:, None])
                ganancia = np.where(valido, ganancia, -np.inf)
                mejor = np.argmax(ganancia, axis=1)
                g_mejor = ganancia[np.arange(Jd), mejor]
                parte = g_mejor > 1e-12
                nodos_p = nodos_d[parte]
                atr = cand[parte, mejor[parte] // r]
                atributo[nodos_p] = atr
                corte[nodos_p] = cortes.reshape(Jd, C)[parte, mejor[parte]]
                rango[nodos_p] = np.arange(nodos_p.size)
                np.add.at(importancias, atr, g_mejor[parte])
            niveles.append({"atributo": atributo, "corte": corte, "rango": rango, "valor": F,
                            "peso": w_nodo, "filas": filas_nodo})
            sigue = rango[j] >= 0
            j, fila, peso = j[sigue], fila[sigue], peso[sigue]
            if j.size:
                derecha = X[fila, atributo[j]] >= corte[j]
                j = 2 * rango[j] + derecha.astype(np.int64)
            J = 2 * int((rango >= 0).sum())
            profundidad += 1
        return niveles, conteos.shape[0]

    def _ensamblar(self, partes):
        atributo, corte, izq, der, valor, peso, filas, raices = [], [], [], [], [], [], [], []
        base = 0
        for niveles, n_arboles in partes:
            tamanos = [len(nv["atributo"]) for nv in niveles]
            desplaz = np.concatenate([[0], np.cumsum(tamanos)]) + base
            raices.append(desplaz[0] + np.arange(n_arboles))
            for d, nv in enumerate(niveles):
                hijo = np.where(nv["rango"] >= 0, desplaz[d + 1] + 2 * nv["rango"], -1)
                atributo.append(nv["atributo"])
                corte.append(nv["corte"])
                izq.append(hijo)
                der.append(np.where(hijo >= 0, hijo + 1, -1))
                valor.append(nv["valor"])
                peso.append(nv["peso"])
                filas.append(nv["filas"])
            base = desplaz[-1]
        self.atributo_ = np.concatenate(atributo).astype(np.int32)
        self.corte_ = np.concatenate(corte)
        self.izquierdo_ = np.concatenate(izq).astype(np.int64)
        self.derecho_ = np.concatenate(der).astype(np.int64)
        self.valor_ = np.concatenate(valor)
        self.peso_nodo_ = np.concatenate(peso)
        self.filas_nodo_ = np.concatenate(filas)
        self.raices_ = np.concatenate(raices).astype(np.int64)
        self.valor_agregado_ = self.valor_ @ self.pesos_agregacion_

    def aplicar(self, X):
        """Índice del nodo hoja de cada muestra en cada árbol, forma (n, n_estimators)."""
        check_is_fitted(self, "raices_")
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=True, reset=False)
        return self._hojas(X)

    def _hojas(self, X):
        nodo = np.tile(self.raices_, (X.shape[0], 1))
        fila = np.repeat(np.arange(X.shape[0]), self.raices_.size)
        nodo = nodo.ravel()
        activo = np.flatnonzero(self.atributo_[nodo] >= 0)
        while activo.size:
            nd = nodo[activo]
            va_izq = X[fila[activo], self.atributo_[nd]] < self.corte_[nd]
            nodo[activo] = np.where(va_izq, self.izquierdo_[nd], self.derecho_[nd])
            activo = activo[self.atributo_[nodo[activo]] >= 0]
        return nodo.reshape(X.shape[0], self.raices_.size)

    def _fuera_de_bolsa(self, X, evento, conteos):
        hojas = self._hojas(X)
        fuera = (conteos.T == 0)
        v = self.valor_agregado_[hojas] * fuera
        suma, cuenta = np.cumsum(v, axis=1), np.cumsum(fuera, axis=1)
        T = self.raices_.size
        puntos = np.unique(np.clip(np.round(np.geomspace(5, T, 15)).astype(int), 1, T))
        curva = []
        for t in puntos:
            ok = cuenta[:, t - 1] > 0
            s = suma[ok, t - 1] / cuenta[ok, t - 1]
            curva.append((int(t), auc_mann_whitney(evento[ok], s)))
        self.curva_oob_ = curva
        ok = cuenta[:, -1] > 0
        self.puntaje_oob_ = np.full(X.shape[0], np.nan)
        self.puntaje_oob_[ok] = suma[ok, -1] / cuenta[ok, -1]
        self.oob_score_ = auc_mann_whitney(evento[ok], self.puntaje_oob_[ok])
        self.calibracion_ = (0.0, 1.0)
        self.historia_calibracion_ = []
        if self.calibrar and ok.sum() > 10 and 0 < evento[ok].sum() < ok.sum():
            z = logit(np.clip(self.puntaje_oob_[ok], 1e-6, 1 - 1e-6))
            a, b, hist = ajustar_platt(z, evento[ok])
            self.historia_calibracion_ = hist
            if b > 1e-6:
                self.calibracion_ = (a, b)

    def predict_distribucion(self, X):
        """Probabilidades acumuladas P(y >= t_k | x) del bosque, forma (n, K)."""
        check_is_fitted(self, "raices_")
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=True, reset=False)
        hojas = self._hojas(X)
        return np.column_stack([self.valor_[hojas, k].mean(axis=1) for k in range(self.umbrales_.size)])

    def predict_niveles(self, X):
        """Probabilidad de cada nivel de y, obtenida por diferencias de las acumuladas.

        Returns
        -------
        niveles : ndarray (L,)
            Nivel más bajo seguido de los umbrales.
        prob : ndarray (n, L)
        """
        acum = self.predict_distribucion(X)
        ext = np.column_stack([np.ones(acum.shape[0]), acum, np.zeros(acum.shape[0])])
        prob = np.clip(ext[:, :-1] - ext[:, 1:], 0.0, None)
        prob /= prob.sum(axis=1, keepdims=True)
        bajos = np.concatenate([[-np.inf], self.umbrales_])
        return bajos, prob

    def decision_function(self, X):
        """Puntaje agregado s(x) = sum_k omega_k P_k(x), en [0, 1]."""
        check_is_fitted(self, "raices_")
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=True, reset=False)
        return self.valor_agregado_[self._hojas(X)].mean(axis=1)

    def predict_proba(self, X):
        """Probabilidad calibrada del evento, columnas [P(Y=0), P(Y=1)]."""
        s = self.decision_function(X)
        a, b = self.calibracion_
        p1 = expit(a + b * logit(np.clip(s, 1e-6, 1 - 1e-6))) if (a, b) != (0.0, 1.0) else s
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """Clase del evento (1 si la probabilidad calibrada es >= 0.5)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y, sample_weight=None):
        """Exactitud frente al evento y >= umbral (y puede ser ordinal o binaria)."""
        check_is_fitted(self, "raices_")
        y = np.asarray(y, dtype=float)
        evento = (y >= self.umbral_evento_).astype(int)
        return float(np.average(self.predict(X) == evento, weights=sample_weight))
