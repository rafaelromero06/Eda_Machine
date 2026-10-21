from __future__ import annotations

import cProfile
import heapq
import io
import pickle
import pstats
import time

import numpy as np
import pandas as pd

from . import config
from .predicciones import construir_modelo

RUTA_COSTO = config.RUTA_EXPERIMENTOS / "costo"

COMPLEJIDAD = pd.DataFrame([
    ("KNN", "O(d·n·log n) (árbol KD) u O(1) (fuerza bruta)", "O(n·d) fuerza bruta; ≈ O(d·log n) con árbol", "O(n·d): guarda los datos", "n (no k)"),
    ("Naive Bayes", "O(n·d)", "O(d)", "O(d)", "—"),
    ("Regresión logística / Ridge / Lasso", "O(n·d·iteraciones)", "O(d)", "O(d)", "C o alpha (iteraciones)"),
    ("Árbol de decisión", "O(d·n·log n)", "O(p)", "O(hojas)", "max_depth, min_samples_leaf"),
    ("Random Forest", "O(T·m·n·log n)", "O(T·p)", "O(T·hojas)", "n_estimators, max_features"),
    ("XGBoost (hist)", "O(T·(n·d + 2ᵖ·b))", "O(T·p)", "O(T·2ᵖ)", "n_estimators, max_depth"),
    ("SVM / SVR (RBF)", "entre O(n²·d) y O(n³·d)", "O(n_sv·d)", "O(n_sv·d)", "C, gamma, n"),
    ("SMOTE / ADASYN", "O(n_min·log n·d) (vecinos con árbol)", "—", "—", "n_min, k = 5"),
], columns=["modelo", "entrenamiento", "predicción por caso", "memoria del modelo", "qué lo encarece"])


def ajustes_validacion_anidada(evaluaciones: int = config.PRESUPUESTO_EVALUACIONES,
                               k_externo: int = config.K_OUTER, k_interno: int = config.K_INNER) -> int:
    """Modelos que se entrenan por combinación: K_ext · (E · K_int + 1)."""
    return k_externo * (evaluaciones * k_interno + 1)


def cronometrar(funcion, repeticiones: int = 3) -> float:
    """Mediana del tiempo de pared (s) de ``funcion()`` en varias repeticiones."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - t0)
    return float(np.median(tiempos))


def remuestrear(X: pd.DataFrame, y: pd.Series, n: int, semilla: int = config.RANDOM_STATE,
                ruido: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
    """Muestra de tamaño ``n`` con la misma proporción de clases (con reemplazo y ruido si ``n`` > datos)."""
    rng = np.random.default_rng(semilla)
    reemplazo = n > len(X)
    if set(np.unique(y)) <= {0, 1}:
        idx = np.concatenate([rng.choice(np.flatnonzero(y.to_numpy() == c), int(round(n * (y == c).mean())),
                                         replace=reemplazo) for c in (0, 1)])
    else:
        idx = rng.choice(len(X), n, replace=reemplazo)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    if reemplazo:
        num = config.COLUMNAS_NUMERICAS
        Xs[num] = Xs[num] + rng.normal(0, 1, (len(Xs), len(num))) * (X[num].std().to_numpy() * ruido)
    return Xs, ys


def escalamiento(tarea: str, modelo: str, balanceo: str, hiperparametros: dict, X: pd.DataFrame, y: pd.Series,
                 tamanos, X_prueba: pd.DataFrame, repeticiones: int = 3) -> pd.DataFrame:
    """Tiempo de entrenamiento y de predicción (de ``X_prueba``) para cada tamaño de muestra."""
    filas = []
    for n in tamanos:
        Xn, yn = remuestrear(X, y, n)
        pipe = construir_modelo(tarea, modelo, balanceo, hiperparametros)
        t_fit = cronometrar(lambda: pipe.fit(Xn, yn), repeticiones)
        predecir = pipe.predict_proba if hasattr(pipe, "predict_proba") and tarea == "clasificacion" else pipe.predict
        t_pred = cronometrar(lambda: predecir(X_prueba), repeticiones)
        filas.append({"modelo": modelo, "balanceo": balanceo, "n": n, "t_fit_s": t_fit, "t_pred_s": t_pred})
    return pd.DataFrame(filas)


def pendiente_loglog(x, y) -> float:
    """Exponente empírico: pendiente de log(y) contra log(x) por mínimos cuadrados."""
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def tiempos_por_etapa(tarea: str, modelo: str, balanceo: str, hiperparametros: dict,
                      X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, repeticiones: int = 3) -> dict:
    """Mediana de los segundos de cada etapa: preprocesar, balancear, ajustar y predecir."""
    from sklearn.base import clone

    pipe = construir_modelo(tarea, modelo, balanceo, hiperparametros)
    pasos = dict(pipe.steps)
    muestreador = next((v for k, v in pasos.items() if k in ("smote", "adasyn")), None)
    medidas = {"preprocesar": [], "balancear": [], "ajustar": [], "predecir": []}
    for _ in range(repeticiones):
        prep, est = clone(pasos["prep"]), clone(pasos["modelo"])
        t0 = time.perf_counter(); Xt = prep.fit_transform(X_tr); t1 = time.perf_counter()
        yt = y_tr
        if muestreador is not None:
            Xt, yt = clone(muestreador).fit_resample(Xt, y_tr)
        t2 = time.perf_counter(); est.fit(Xt, yt); t3 = time.perf_counter()
        Xp = prep.transform(X_te)
        if tarea == "clasificacion" and hasattr(est, "predict_proba"):
            est.predict_proba(Xp)
        else:
            est.predict(Xp)
        t4 = time.perf_counter()
        for etapa, valor in zip(medidas, [t1 - t0, t2 - t1, t3 - t2, t4 - t3]):
            medidas[etapa].append(valor)
    return {"modelo": modelo, "balanceo": balanceo, **{k: float(np.median(v)) for k, v in medidas.items()}}


def perfilar(funcion, top: int | None = 15) -> tuple[pd.DataFrame, float]:
    """Perfil de ``funcion()`` con cProfile.

    Returns
    -------
    (DataFrame, float)
        Las ``top`` funciones con más tiempo propio (todas si ``top`` es None):
        llamadas, tiempo propio y acumulado, archivo; y el tiempo total del perfil.
    """
    perfil = cProfile.Profile()
    t0 = time.perf_counter()
    perfil.runcall(funcion)
    total = time.perf_counter() - t0
    estadisticas = pstats.Stats(perfil, stream=io.StringIO())
    filas = []
    for (archivo, linea, nombre), (cc, nc, tt, ct, _) in estadisticas.stats.items():
        filas.append({"funcion": nombre, "archivo": _abreviar(archivo), "linea": linea, "llamadas": nc,
                      "tiempo_propio_s": tt, "tiempo_acumulado_s": ct})
    tabla = pd.DataFrame(filas).sort_values("tiempo_propio_s", ascending=False)
    tabla = (tabla if top is None else tabla.head(top)).reset_index(drop=True)
    tabla["%_del_total"] = 100 * tabla["tiempo_propio_s"] / total
    return tabla, total


def _abreviar(archivo: str) -> str:
    """Ruta de un módulo desde 'site-packages' (o el nombre si es una función interna)."""
    archivo = archivo.replace("\\", "/")
    if "site-packages/" in archivo:
        return archivo.split("site-packages/")[-1]
    if archivo.startswith("~") or archivo.startswith("<"):
        return "(interna de Python/C)"
    return archivo.split("/")[-1]


def planificar(duraciones, n_procesos: int) -> float:
    """Makespan de una cola de tareas: cada tarea va al primer proceso libre, en el orden dado."""
    libres = [0.0] * n_procesos
    heapq.heapify(libres)
    for d in duraciones:
        heapq.heappush(libres, heapq.heappop(libres) + float(d))
    return max(libres)


def tamano_en_memoria(estimador) -> float:
    """Tamaño del modelo serializado con pickle, en MB."""
    return len(pickle.dumps(estimador, protocol=pickle.HIGHEST_PROTOCOL)) / 1024 ** 2
