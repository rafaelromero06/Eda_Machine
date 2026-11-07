from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .benchmarks import NOMBRE_ORIGINAL, construir_pipeline, puntaje, usa_respuesta_ordinal
from .data import NUMERICAS_VINO, cargar_conjunto
from .validacion import particiones

SEMILLA = 42
GRILLAS_SENSIBILIDAD = {
    "ancho_agregacion": [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 5.0, np.inf],
    "ancho_particion": [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf],
    "desplazamiento": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    "max_features": [1, 2, "sqrt", 5, 8, None],
    "min_samples_leaf": [1, 2, 3, 5, 10, 20],
    "n_estimators": [25, 50, 100, 200, 300, 500],
}
ESCENARIOS_RUIDO = [("variables", 0.1), ("variables", 0.25), ("variables", 0.5), ("etiqueta", 0.05), ("etiqueta", 0.10),
                    ("etiqueta", 0.20), ("atipicos", 0.01), ("atipicos", 0.05)]
FRACCIONES = [0.10, 0.25, 0.50, 1.0]
SEMILLAS_MODELO = list(range(10))
ALFAS = [0.10, 0.05]


def hiperparametros_finales(ruta):
    tabla = pd.read_csv(ruta)
    salida = {}
    for r in tabla.itertuples():
        hp = json.loads(r.hiperparametros)
        salida[r.modelo] = {k: (float(v) if isinstance(v, str) and v in ("inf", "-inf") else v) for k, v in hp.items()}
    return salida


def _parametros_semilla(modelo, semilla):
    if modelo == "CatBoost":
        return {"modelo__random_seed": semilla}
    if modelo in ("KNN", "Naive Bayes"):
        return {}
    cambios = {"modelo__random_state": semilla}
    if modelo == "XGBoost":
        cambios["smote__random_state"] = semilla
    return cambios


def ajustar(modelo, hp, X_tr, yord_tr, yev_tr, datos, cambios=None, semilla=None):
    """Ajusta el pipeline final de un modelo con hiperparámetros fijos y devuelve (pipeline, segundos)."""
    warnings.filterwarnings("ignore")
    pipe = construir_pipeline(modelo, datos)
    pipe.set_params(**{f"modelo__{k}": v for k, v in hp.items()})
    if cambios:
        pipe.set_params(**{f"modelo__{k}": v for k, v in cambios.items()})
    if semilla is not None:
        pipe.set_params(**_parametros_semilla(modelo, semilla))
    t0 = time.perf_counter()
    pipe.fit(X_tr, yord_tr if usa_respuesta_ordinal(modelo) else yev_tr)
    return pipe, time.perf_counter() - t0


def _contaminar(estudio, nivel, X_tr, X_te, yord_tr, rng):
    X_tr, X_te, yord_tr = X_tr.copy(), X_te.copy(), yord_tr.copy()
    if estudio == "variables":
        sd = X_tr[NUMERICAS_VINO].std()
        X_tr[NUMERICAS_VINO] += rng.normal(size=(len(X_tr), len(NUMERICAS_VINO))) * (nivel * sd.to_numpy())
        X_te[NUMERICAS_VINO] += rng.normal(size=(len(X_te), len(NUMERICAS_VINO))) * (nivel * sd.to_numpy())
    elif estudio == "etiqueta":
        idx = rng.choice(len(yord_tr), int(round(nivel * len(yord_tr))), replace=False)
        yord_tr[idx] = np.clip(yord_tr[idx] + rng.choice([-1, 1], idx.size), 3, 9)
    elif estudio == "atipicos":
        idx = rng.choice(len(X_tr), int(round(nivel * len(X_tr))), replace=False)
        cols = rng.choice(len(NUMERICAS_VINO), idx.size)
        for i, c in zip(idx, cols):
            X_tr.iloc[i, X_tr.columns.get_loc(NUMERICAS_VINO[c])] *= 10.0
    return X_tr, X_te, yord_tr


def tarea(estudio, modelo, fold, hp, parametro=None, valor=None):
    """Una evaluación de robustez sobre un pliegue externo de la repetición 1 (semilla 42).

    Returns
    -------
    list de dict, una fila por resultado.
    """
    datos = cargar_conjunto("vino")
    X, y_ord, y_ev = datos["X"], datos["y_ord"], datos["y_evento"]
    tr, te = particiones(y_ev, SEMILLA)[fold]
    X_tr, X_te, yord_tr = X.iloc[tr], X.iloc[te], y_ord[tr]
    base = {"estudio": estudio, "modelo": modelo, "fold": fold, "parametro": parametro, "valor": str(valor)}
    rng = np.random.default_rng(1000 + fold)
    if estudio == "referencia":
        pipe, seg = ajustar(modelo, hp, X_tr, yord_tr, y_ev[tr], datos)
        return [{**base, "auc": roc_auc_score(y_ev[te], puntaje(pipe, X_te)), "segundos": seg}]
    if estudio == "sensibilidad":
        pipe, seg = ajustar(modelo, hp, X_tr, yord_tr, y_ev[tr], datos, cambios={parametro: valor})
        return [{**base, "auc": roc_auc_score(y_ev[te], puntaje(pipe, X_te)), "segundos": seg}]
    if estudio == "semillas":
        pipe, seg = ajustar(modelo, hp, X_tr, yord_tr, y_ev[tr], datos, semilla=valor)
        return [{**base, "auc": roc_auc_score(y_ev[te], puntaje(pipe, X_te)), "segundos": seg}]
    if estudio == "ruido":
        X_tr2, X_te2, yord2 = _contaminar(parametro, valor, X_tr, X_te, yord_tr, rng)
        pipe, seg = ajustar(modelo, hp, X_tr2, yord2, (yord2 >= datos["umbral"]).astype(int), datos)
        return [{**base, "auc": roc_auc_score(y_ev[te], puntaje(pipe, X_te2)), "segundos": seg}]
    if estudio == "tamano":
        if valor < 1.0:
            sub, _ = train_test_split(np.arange(len(tr)), train_size=valor, stratify=y_ev[tr], random_state=SEMILLA)
        else:
            sub = np.arange(len(tr))
        pipe, seg = ajustar(modelo, hp, X_tr.iloc[sub], yord_tr[sub], y_ev[tr][sub], datos)
        return [{**base, "n_entrenamiento": len(sub), "auc": roc_auc_score(y_ev[te], puntaje(pipe, X_te)), "segundos": seg}]
    if estudio == "conforme":
        return _conforme(modelo, hp, X_tr, X_te, yord_tr, y_ev[tr], y_ev[te], datos, base)
    raise ValueError(estudio)


def _conforme(modelo, hp, X_tr, X_te, yord_tr, yev_tr, yev_te, datos, base):
    aj, cal = train_test_split(np.arange(len(X_tr)), test_size=0.25, stratify=yev_tr, random_state=SEMILLA)
    pipe, _ = ajustar(modelo, hp, X_tr.iloc[aj], yord_tr[aj], yev_tr[aj], datos)
    s_cal, s_te = puntaje(pipe, X_tr.iloc[cal]), puntaje(pipe, X_te)
    if modelo == "SVM":
        s_cal, s_te = 1 / (1 + np.exp(-s_cal)), 1 / (1 + np.exp(-s_te))
    y_cal = yev_tr[cal]
    no_conf_cal = np.where(y_cal == 1, 1 - s_cal, s_cal)
    filas = []
    for alfa in ALFAS:
        n = len(no_conf_cal)
        q = np.quantile(no_conf_cal, min(1.0, np.ceil((n + 1) * (1 - alfa)) / n), method="higher")
        q_clase = {}
        for c in (0, 1):
            nc = no_conf_cal[y_cal == c]
            q_clase[c] = np.quantile(nc, min(1.0, np.ceil((nc.size + 1) * (1 - alfa)) / nc.size), method="higher")
        for tipo, (q0, q1) in {"marginal": (q, q), "por_clase": (q_clase[0], q_clase[1])}.items():
            incluye1 = (1 - s_te) <= q1
            incluye0 = s_te <= q0
            cubre = np.where(yev_te == 1, incluye1, incluye0)
            tam = incluye0.astype(int) + incluye1.astype(int)
            for clase in (0, 1):
                m = yev_te == clase
                filas.append({**base, "alfa": alfa, "tipo": tipo, "clase": clase, "n": int(m.sum()),
                              "cobertura": float(cubre[m].mean()), "tamano_medio": float(tam[m].mean()),
                              "vacios": float((tam[m] == 0).mean()), "dobles": float((tam[m] == 2).mean())})
    return filas


def cambio_distribucion(modelo, hp, origen):
    """Entrena con un tipo de vino y evalúa en el otro."""
    datos = cargar_conjunto("vino")
    X, y_ord, y_ev = datos["X"], datos["y_ord"], datos["y_evento"]
    a = (X["tipo_vino"] == origen).to_numpy()
    pipe, seg = ajustar(modelo, hp, X[a], y_ord[a], y_ev[a], datos)
    return [{"estudio": "cambio_distribucion", "modelo": modelo, "entrena": origen,
             "evalua": "white" if origen == "red" else "red", "n_entrenamiento": int(a.sum()),
             "auc": roc_auc_score(y_ev[~a], puntaje(pipe, X[~a])), "segundos": seg}]


def escalabilidad(modelo, hp, repeticiones=3):
    """Tiempo de ajuste y de predicción al variar n (filas) y p (variables de ruido agregadas)."""
    datos = cargar_conjunto("vino")
    X, y_ord, y_ev = datos["X"], datos["y_ord"], datos["y_evento"]
    rng = np.random.default_rng(SEMILLA)
    filas = []
    _, prueba = ajustar(modelo, hp, X, y_ord, y_ev, datos)
    if prueba > 30:
        repeticiones = 1
    for n in [500, 1000, 2000, 4000, len(y_ev)]:
        idx = rng.choice(len(y_ev), n, replace=False)
        for _ in range(repeticiones):
            pipe, seg = ajustar(modelo, hp, X.iloc[idx], y_ord[idx], y_ev[idx], datos)
            t0 = time.perf_counter()
            puntaje(pipe, X.iloc[:1000])
            filas.append({"modelo": modelo, "eje": "n", "n": n, "p": 13, "ajuste_s": seg,
                          "prediccion_ms_por_caso": (time.perf_counter() - t0)})
    for extra in [0, 13, 39]:
        Xp = X.copy()
        nuevas = {f"ruido_{j}": rng.normal(size=len(Xp)) for j in range(extra)}
        Xp = pd.concat([Xp, pd.DataFrame(nuevas, index=Xp.index)], axis=1) if extra else Xp
        datos_p = {**datos, "numericas": datos["numericas"] + list(nuevas)}
        for _ in range(repeticiones):
            pipe, seg = ajustar(modelo, hp, Xp, y_ord, y_ev, datos_p)
            t0 = time.perf_counter()
            puntaje(pipe, Xp.iloc[:1000])
            filas.append({"modelo": modelo, "eje": "p", "n": len(y_ev), "p": 13 + extra, "ajuste_s": seg,
                          "prediccion_ms_por_caso": (time.perf_counter() - t0)})
    return filas


def validacion_por_bloques(modelo, hp, tamano_bloque=10):
    """Validación con pliegues formados por bloques de vinos consecutivos del archivo original (posibles lotes)."""
    from sklearn.model_selection import GroupKFold
    datos = cargar_conjunto("vino")
    X, y_ord, y_ev = datos["X"], datos["y_ord"], datos["y_evento"]
    posicion = X.groupby("tipo_vino").cumcount().to_numpy()
    grupos = np.where(X["tipo_vino"].to_numpy() == "red", 0, 100000) + posicion // tamano_bloque
    filas = []
    for fold, (tr, te) in enumerate(GroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y_ev, grupos)):
        pipe, seg = ajustar(modelo, hp, X.iloc[tr], y_ord[tr], y_ev[tr], datos)
        filas.append({"estudio": "bloques_archivo", "modelo": modelo, "fold": fold, "parametro": "tamano_bloque",
                      "valor": str(tamano_bloque), "auc": roc_auc_score(y_ev[te], puntaje(pipe, X.iloc[te])), "segundos": seg})
    return filas
