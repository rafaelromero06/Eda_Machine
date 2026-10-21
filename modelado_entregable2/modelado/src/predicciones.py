from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold

from . import config
from .pipeline_builder import construir_pipeline

RUTA_PREDICCIONES = config.RUTA_EXPERIMENTOS / "predicciones"
CLAVES = ["modelo", "balanceo", "optimizador"]
_ESCALAR_NUMPY = re.compile(r"np\.(?:float|int|bool|str)_?\d*\((.*?)\)")

_PESO_MODELO = {"Random Forest": 10, "XGBoost": 5, "SVM": 3, "SVR": 3, "KNN": 1.5}
_PESO_BALANCEO = {"SMOTE": 2, "ADASYN": 2}


def parsear_hiperparametros(texto: str) -> dict:
    """Convierte la columna ``mejores_hiperparametros`` del log en un diccionario.

    Parameters
    ----------
    texto : str
        Representación de texto del diccionario, por ejemplo
        ``"{'C': np.float64(3.40), 'penalty': 'l1'}"``.

    Returns
    -------
    dict
        Hiperparámetros con tipos nativos de Python.
    """
    return ast.literal_eval(_ESCALAR_NUMPY.sub(r"\1", texto))


def particiones_externas(tarea: str, X: pd.DataFrame, y: pd.Series) -> list:
    """Particiones del bucle externo, idénticas a las de ``nested_cv``.

    Parameters
    ----------
    tarea : {"clasificacion", "regresion"}
    X, y : datos completos (índice 0..n-1).

    Returns
    -------
    list of (ndarray, ndarray)
        Índices de entrenamiento y prueba de cada fold externo.
    """
    cv = StratifiedKFold if tarea == "clasificacion" else KFold
    divisor = cv(n_splits=config.K_OUTER, shuffle=True, random_state=config.RANDOM_STATE)
    return list(divisor.split(X, y))


def construir_modelo(tarea: str, modelo: str, balanceo: str, hiperparametros: dict,
                     calibracion: str | None = None):
    """Pipeline con los hiperparámetros elegidos, opcionalmente recalibrado.

    Parameters
    ----------
    tarea, modelo, balanceo : identificación de la combinación.
    hiperparametros : dict
        Hiperparámetros del modelo (sin el prefijo ``modelo__``).
    calibracion : {None, "sigmoid", "isotonic"}
        Si se indica, el pipeline se envuelve en ``CalibratedClassifierCV`` con
        3 folds estratificados sobre el entrenamiento (Platt o isotónica).

    Returns
    -------
    estimator
        Estimador sin entrenar.
    """
    pipe, aplica = construir_pipeline(tarea, modelo, balanceo)
    if not aplica:
        raise ValueError(f"La combinación {modelo} × {balanceo} no aplica")
    pipe.set_params(**{f"modelo__{k}": v for k, v in hiperparametros.items()})
    if calibracion is None:
        return pipe
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    return CalibratedClassifierCV(pipe, method=calibracion, cv=cv)


def puntaje_clasificacion(estimador, X) -> tuple[np.ndarray, str]:
    """Puntaje continuo de la clase positiva y su tipo.

    Returns
    -------
    (ndarray, str)
        Probabilidad (``"probabilidad"``) si el estimador tiene
        ``predict_proba``; si no (SVM), la función de decisión (``"decision"``).
    """
    if hasattr(estimador, "predict_proba"):
        return estimador.predict_proba(X)[:, 1], "probabilidad"
    return estimador.decision_function(X), "decision"


def predecir_fold(tarea: str, modelo: str, balanceo: str, optimizador: str, fold: int,
                  hiperparametros: dict, X: pd.DataFrame, y: pd.Series,
                  calibracion: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Reentrena un fold externo y predice su prueba externa.

    Parameters
    ----------
    tarea, modelo, balanceo, optimizador : identificación de la combinación.
    fold : int
        Fold externo (0..K_OUTER-1).
    hiperparametros : dict
        Configuración elegida por la búsqueda en ese fold.
    X, y : datos completos.
    calibracion : {None, "sigmoid", "isotonic"}
        Recalibración de probabilidades (solo clasificación).

    Returns
    -------
    (DataFrame, dict)
        Predicciones de la prueba externa (``idx``, ``y_true``, ``score``/``pred``)
        y tiempos de entrenamiento y predicción en segundos.
    """
    import warnings
    warnings.filterwarnings("ignore")
    tr, te = particiones_externas(tarea, X, y)[fold]
    est = construir_modelo(tarea, modelo, balanceo, hiperparametros, calibracion)
    t0 = time.perf_counter()
    est.fit(X.iloc[tr], y.iloc[tr])
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    salida = pd.DataFrame({"modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
                           "fold": fold, "idx": te, "y_true": y.iloc[te].to_numpy()})
    if tarea == "clasificacion":
        score, tipo = puntaje_clasificacion(est, X.iloc[te])
        salida["score"] = score
        salida["pred"] = est.predict(X.iloc[te]).astype(int)
        salida["tipo_score"] = tipo
    else:
        salida["pred"] = est.predict(X.iloc[te])
    t_pred = time.perf_counter() - t0
    return salida, {"modelo": modelo, "balanceo": balanceo, "optimizador": optimizador, "fold": fold,
                    "calibracion": calibracion or "ninguna", "n_train": len(tr), "n_test": len(te),
                    "t_fit_s": t_fit, "t_pred_s": t_pred}


def tareas_desde_log(log: pd.DataFrame) -> list[dict]:
    """Lista de tareas (una por fila del log), de la más pesada a la más liviana."""
    tareas = [{"modelo": r.modelo, "balanceo": r.balanceo, "optimizador": r.optimizador,
               "fold": int(r.fold_externo), "hiperparametros": parsear_hiperparametros(r.mejores_hiperparametros)}
              for r in log.itertuples()]
    tareas.sort(key=lambda t: _PESO_MODELO.get(t["modelo"], 1) * _PESO_BALANCEO.get(t["balanceo"], 1), reverse=True)
    return tareas


def generar_oof(tarea: str, log: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                n_workers: int | None = None, calibracion: str | None = None,
                verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reentrena en paralelo todas las filas de ``log`` y junta las predicciones.

    Parameters
    ----------
    tarea : {"clasificacion", "regresion"}
    log : DataFrame
        Filas del log a reconstruir (todas o un subconjunto).
    X, y : datos completos.
    n_workers : int, optional
        Procesos en paralelo (por defecto ``config.N_WORKERS``).
    calibracion : {None, "sigmoid", "isotonic"}

    Returns
    -------
    (DataFrame, DataFrame)
        Predicciones OOF en formato largo y tiempos por tarea.
    """
    from joblib import Parallel, delayed

    n_workers = n_workers or config.N_WORKERS
    tareas = tareas_desde_log(log)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    trabajos = (delayed(predecir_fold)(tarea, t["modelo"], t["balanceo"], t["optimizador"], t["fold"],
                                       t["hiperparametros"], X, y, calibracion) for t in tareas)
    predicciones, tiempos = [], []
    t0 = time.time()
    for i, (pred, meta) in enumerate(Parallel(n_jobs=n_workers, return_as="generator_unordered",
                                              batch_size=1)(trabajos), start=1):
        predicciones.append(pred)
        tiempos.append(meta)
        if verbose and (i % 25 == 0 or i == len(tareas)):
            print(f"  {tarea}: {i}/{len(tareas)} tareas | {time.time() - t0:5.0f} s", flush=True)
    oof = pd.concat(predicciones, ignore_index=True).sort_values(CLAVES + ["idx"]).reset_index(drop=True)
    return oof, pd.DataFrame(tiempos)


def umbral_fold(modelo: str, balanceo: str, optimizador: str, fold: int, hiperparametros: dict,
                X: pd.DataFrame, y: pd.Series, umbrales=None) -> dict:
    """Umbral con mayor F1 en predicciones internas del entrenamiento externo de un fold.

    Returns
    -------
    dict
        Combinación, fold, umbral elegido y F1 interno con ese umbral y con 0.5.
    """
    import warnings
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict
    warnings.filterwarnings("ignore")
    umbrales = np.round(np.arange(0.05, 0.951, 0.01), 2) if umbrales is None else np.asarray(umbrales)
    tr, _ = particiones_externas("clasificacion", X, y)[fold]
    est = construir_modelo("clasificacion", modelo, balanceo, hiperparametros)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    prob = cross_val_predict(est, X.iloc[tr], y.iloc[tr], cv=cv, method="predict_proba")[:, 1]
    f1 = np.array([f1_score(y.iloc[tr], prob >= u, zero_division=0) for u in umbrales])
    mejor = int(np.argmax(f1))
    return {"modelo": modelo, "balanceo": balanceo, "optimizador": optimizador, "fold": fold,
            "umbral": float(umbrales[mejor]), "f1_interno": float(f1[mejor]),
            "f1_interno_05": float(f1[np.argmin(np.abs(umbrales - 0.5))])}


def umbrales_desde_log(log: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                       n_workers: int | None = None) -> pd.DataFrame:
    """``umbral_fold`` en paralelo para todas las filas (combinación × fold) de ``log``."""
    from joblib import Parallel, delayed

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    trabajos = (delayed(umbral_fold)(t["modelo"], t["balanceo"], t["optimizador"], t["fold"],
                                     t["hiperparametros"], X, y) for t in tareas_desde_log(log))
    filas = Parallel(n_jobs=n_workers or config.N_WORKERS, batch_size=1)(trabajos)
    return pd.DataFrame(filas).sort_values(CLAVES + ["fold"]).reset_index(drop=True)


def ruta_oof(tarea: str, calibracion: str | None = None) -> Path:
    sufijo = "" if calibracion is None else f"_{calibracion}"
    return RUTA_PREDICCIONES / f"oof_{tarea}{sufijo}.csv.gz"


def cargar_oof(tarea: str, calibracion: str | None = None) -> pd.DataFrame:
    """Lee las predicciones OOF guardadas por ``generar_predicciones.py``."""
    return pd.read_csv(ruta_oof(tarea, calibracion), float_precision="round_trip")


def metricas_por_fold(oof: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Métricas de cada combinación y fold calculadas desde las predicciones OOF."""
    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                                 precision_score, recall_score, roc_auc_score)

    filas = []
    for clave, g in oof.groupby(CLAVES + ["fold"], sort=False):
        fila = dict(zip(CLAVES + ["fold_externo"], clave))
        if tarea == "clasificacion":
            fila.update(auc=roc_auc_score(g.y_true, g.score), accuracy=accuracy_score(g.y_true, g.pred),
                        precision=precision_score(g.y_true, g.pred, zero_division=0),
                        recall=recall_score(g.y_true, g.pred, zero_division=0),
                        f1=f1_score(g.y_true, g.pred, zero_division=0))
        else:
            fila.update(rmse=float(mean_squared_error(g.y_true, g.pred) ** 0.5),
                        mae=float(mean_absolute_error(g.y_true, g.pred)))
        filas.append(fila)
    return pd.DataFrame(filas)
