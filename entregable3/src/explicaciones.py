from __future__ import annotations

import copy

import numpy as np
import pandas as pd

NOMBRE_ORIGINAL = "Bosque Ordinal Focalizado"
TAMANO_BLOQUE = 25


def nombres_originales(prep):
    """Variable original de cada columna transformada (el one-hot del tipo de vino vuelve a una sola variable)."""
    salida = []
    for n in prep.get_feature_names_out():
        base = n.split("__", 1)[1]
        salida.append("tipo_vino" if base.startswith("tipo_vino") else base)
    return np.array(salida)


def indices_bof(bof):
    """Orden de los nodos agrupados por árbol, inicio de cada árbol en ese orden e índice local de cada nodo."""
    n = bof.atributo_.size
    arbol = np.full(n, -1, dtype=np.int32)
    arbol[bof.raices_] = np.arange(bof.raices_.size, dtype=np.int32)
    frente = bof.raices_.astype(np.int64)
    while frente.size:
        internos = frente[bof.atributo_[frente] >= 0]
        hijos_i, hijos_d = bof.izquierdo_[internos].astype(np.int64), bof.derecho_[internos].astype(np.int64)
        arbol[hijos_i] = arbol[internos]
        arbol[hijos_d] = arbol[internos]
        frente = np.concatenate([hijos_i, hijos_d])
    orden = np.argsort(arbol, kind="stable")
    cortes = np.searchsorted(arbol[orden], np.arange(bof.raices_.size + 1))
    local = np.empty(n, dtype=np.int32)
    for t in range(bof.raices_.size):
        local[orden[cortes[t]:cortes[t + 1]]] = np.arange(cortes[t + 1] - cortes[t], dtype=np.int32)
    return orden, cortes, local


def arboles_bof(bof, desde=0, hasta=None, indices=None):
    """Árboles ``desde``..``hasta`` del bosque propio en el formato de diccionarios de shap.TreeExplainer.

    Los valores se dividen por el total de árboles: la suma de las explicaciones de todos los bloques es la del bosque.
    """
    orden, cortes, local = indices_bof(bof) if indices is None else indices
    T = bof.raices_.size
    arboles = []
    for t in range(desde, T if hasta is None else min(hasta, T)):
        ids = orden[cortes[t]:cortes[t + 1]]
        interno = bof.atributo_[ids] >= 0
        izq = np.where(interno, local[np.maximum(bof.izquierdo_[ids], 0)], -1).astype(np.int32)
        der = np.where(interno, local[np.maximum(bof.derecho_[ids], 0)], -1).astype(np.int32)
        arboles.append({"children_left": izq, "children_right": der, "children_default": izq,
                        "features": bof.atributo_[ids].astype(np.int32),
                        "thresholds": np.nan_to_num(bof.corte_[ids].astype(np.float64)),
                        "values": (bof.valor_agregado_[ids].astype(np.float64) / T)[:, None],
                        "node_sample_weight": bof.peso_nodo_[ids].astype(np.float64)})
    return {"trees": arboles, "base_offset": 0.0, "tree_output": "raw_value", "input_dtype": np.float64,
            "internal_dtype": np.float64}


def _shap_arbol(explicador, Z):
    """Valores SHAP de la clase positiva y valor base de un shap.TreeExplainer."""
    v = explicador.shap_values(Z)
    ev = np.ravel(explicador.expected_value)
    if isinstance(v, list):
        return np.asarray(v[1]), float(ev[1])
    v = np.asarray(v)
    if v.ndim == 3:
        return v[:, :, 1], float(ev[1] if ev.size > 1 else ev[0])
    return v, float(ev[-1])


class Explicador:
    """Explicación SHAP de un pipeline final, devuelta por variable original.

    Los bosques (BOF y Random Forest) se explican por bloques de árboles: por linealidad, el SHAP del bosque es la suma
    del de sus bloques, así que el resultado es exacto y la memoria queda acotada. SVM, KNN y Naive Bayes usan valores de
    Shapley por permutaciones antitéticas de las variables originales.
    """

    def __init__(self, nombre_modelo, pipe, fondo, permutaciones=10):
        self.nombre = nombre_modelo
        self.prep = pipe.named_steps["prep"]
        self.modelo = pipe.named_steps["modelo"]
        self.nombres = nombres_originales(self.prep)
        self.permutaciones = permutaciones
        self._arbol = None
        Xf = np.asarray(self.prep.transform(fondo), dtype=float)
        tipo = type(self.modelo).__name__
        if nombre_modelo == NOMBRE_ORIGINAL:
            self.modo, self.escala = "bosque_propio", "puntaje agregado del bosque"
        elif tipo == "RandomForestClassifier":
            self.modo, self.escala = "bosque_sklearn", "probabilidad de vino bueno"
        elif tipo == "DecisionTreeClassifier":
            self.modo, self.escala = "arbol", "probabilidad de vino bueno"
        elif "XGB" in tipo or tipo in ("LGBMClassifier", "CatBoostEnvoltorio"):
            self.modo, self.escala = "arbol", "logit (log-odds) de vino bueno"
        elif tipo == "LogisticRegression":
            self.coef = self.modelo.coef_.ravel()
            self.media = Xf.mean(axis=0)
            self.base = float(self.modelo.intercept_[0] + self.coef @ self.media)
            self.modo, self.escala = "lineal", "logit (log-odds) de vino bueno"
        else:
            self.fondo = Xf[np.random.default_rng(0).choice(len(Xf), min(30, len(Xf)), replace=False)]
            self.modo = "permutacion"
            self.escala = "probabilidad de vino bueno" if hasattr(self.modelo, "predict_proba") else "función de decisión"

    def salida(self, X):
        """Salida del modelo en la escala de la explicación: la suma de los aportes más el valor base la reproduce."""
        return self._salida(np.asarray(self.prep.transform(X), dtype=float))

    def _salida(self, Z):
        m, tipo = self.modelo, type(self.modelo).__name__
        if self.modo == "bosque_propio":
            return m.decision_function(Z)
        if tipo == "CatBoostEnvoltorio":
            return m.modelo_.predict(Z, prediction_type="RawFormulaVal")
        if tipo == "LGBMClassifier":
            return m.predict(Z, raw_score=True)
        if "XGB" in tipo:
            return m.predict(Z, output_margin=True)
        if self.modo == "lineal" or not hasattr(m, "predict_proba"):
            return m.decision_function(Z)
        return m.predict_proba(Z)[:, 1]

    def explicar(self, X):
        """Aportes por variable original de la primera fila de X y valor base."""
        tabla, base = self.explicar_varios(X.iloc[:1])
        return tabla.iloc[0], base

    def explicar_varios(self, X):
        """Aportes por variable original de cada fila de X (DataFrame) y valor base."""
        Z = np.asarray(self.prep.transform(X), dtype=float)
        if self.modo == "lineal":
            valores, base, columnas = self.coef * (Z - self.media), self.base, self.nombres
        elif self.modo == "permutacion":
            columnas = np.array(list(dict.fromkeys(self.nombres)))
            pares = [self._permutacion(z, columnas) for z in Z]
            valores, base = np.vstack([v for v, _ in pares]), pares[0][1]
        else:
            valores, base = self._bosque_propio(Z) if self.modo == "bosque_propio" else (
                self._bosque_sklearn(Z) if self.modo == "bosque_sklearn" else _shap_arbol(self._explicador_arbol(), Z))
            columnas = self.nombres
        tabla = pd.DataFrame(valores, columns=columnas, index=X.index)
        return tabla.T.groupby(level=0, sort=False).sum().T, base

    def _explicador_arbol(self):
        import shap
        if self._arbol is None:
            modelo = self.modelo.modelo_ if type(self.modelo).__name__ == "CatBoostEnvoltorio" else self.modelo
            self._arbol = shap.TreeExplainer(modelo)
        return self._arbol

    def _bosque_propio(self, Z):
        import shap
        indices = indices_bof(self.modelo)
        valores, base = np.zeros(Z.shape), 0.0
        for desde in range(0, self.modelo.raices_.size, TAMANO_BLOQUE):
            v, b = _shap_arbol(shap.TreeExplainer(arboles_bof(self.modelo, desde, desde + TAMANO_BLOQUE, indices)), Z)
            valores += v
            base += b
        return valores, base

    def _bosque_sklearn(self, Z):
        import shap
        T = len(self.modelo.estimators_)
        valores, base = np.zeros(Z.shape), 0.0
        for desde in range(0, T, TAMANO_BLOQUE):
            parte = copy.copy(self.modelo)
            parte.estimators_ = self.modelo.estimators_[desde:desde + TAMANO_BLOQUE]
            v, b = _shap_arbol(shap.TreeExplainer(parte), Z)
            peso = len(parte.estimators_) / T
            valores += peso * v
            base += peso * b
        return valores, base

    def _permutacion(self, z, columnas):
        """Shapley por permutaciones antitéticas de las variables originales; los aportes suman salida menos base."""
        grupos = [np.flatnonzero(self.nombres == c) for c in columnas]
        rng = np.random.default_rng(0)
        k, m = len(grupos), len(self.fondo)
        phi = np.zeros(k)
        for _ in range(self.permutaciones // 2):
            orden = rng.permutation(k)
            for o in (orden, orden[::-1]):
                Zs = np.repeat(self.fondo[None], k + 1, axis=0)
                for paso, g in enumerate(o, 1):
                    Zs[paso:, :, grupos[g]] = z[grupos[g]]
                y = self._salida(Zs.reshape(-1, z.size)).reshape(k + 1, m).mean(axis=1)
                phi[o] += np.diff(y)
        return phi / (2 * (self.permutaciones // 2)), float(self._salida(self.fondo).mean())
