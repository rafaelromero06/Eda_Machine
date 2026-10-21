from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .predicciones import construir_modelo, parsear_hiperparametros, particiones_externas

RUTA_INTERPRETABILIDAD = config.RUTA_EXPERIMENTOS / "interpretabilidad"
VARIABLES = config.COLUMNAS_NUMERICAS + ["tipo_vino"]
TIPOS = ["red", "white"]


def filas_combinacion(log: pd.DataFrame, modelo: str, balanceo: str, optimizador: str) -> pd.DataFrame:
    """Las 5 filas (folds externos) de una combinación del log."""
    filas = log[(log["modelo"] == modelo) & (log["balanceo"] == balanceo) & (log["optimizador"] == optimizador)]
    return filas.sort_values("fold_externo").reset_index(drop=True)


def ajustar_fold(tarea: str, fila, X: pd.DataFrame, y: pd.Series):
    """Pipeline de un fold externo con los hiperparámetros del log, ya entrenado.

    Returns
    -------
    (pipeline, ndarray, ndarray)
        Pipeline entrenado e índices de entrenamiento y prueba del fold.
    """
    tr, te = particiones_externas(tarea, X, y)[int(fila["fold_externo"])]
    pipe = construir_modelo(tarea, fila["modelo"], fila["balanceo"], parsear_hiperparametros(fila["mejores_hiperparametros"]))
    pipe.fit(X.iloc[tr], y.iloc[tr])
    return pipe, tr, te


def _agrupar_tipo(valores: np.ndarray) -> np.ndarray:
    """Suma las columnas del one-hot de tipo_vino (las dos últimas)."""
    n_num = len(config.COLUMNAS_NUMERICAS)
    return np.column_stack([valores[:, :n_num], valores[:, n_num:].sum(axis=1)])


def _shap_lote(tarea: str, fila: dict, X: pd.DataFrame, y: pd.Series, posiciones: np.ndarray):
    """SHAP de una parte de la prueba externa de un fold (se ejecuta en un proceso aparte)."""
    import warnings
    warnings.filterwarnings("ignore")
    import shap

    pipe, _, te = ajustar_fold(tarea, fila, X, y)
    idx = te[posiciones]
    Xt = pipe.named_steps["prep"].transform(X.iloc[idx])
    explicador = shap.TreeExplainer(pipe.named_steps["modelo"])
    t0 = time.perf_counter()
    valores = np.asarray(explicador.shap_values(Xt))
    segundos = time.perf_counter() - t0
    base = np.atleast_1d(explicador.expected_value)
    if valores.ndim == 3:
        valores, base = valores[..., 1], base[1]
    else:
        base = base[0]
    salida = pd.DataFrame(_agrupar_tipo(valores), columns=VARIABLES)
    salida.insert(0, "idx", idx)
    salida.insert(1, "fold", int(fila["fold_externo"]))
    salida["base"] = float(base)
    return salida, {"fold": int(fila["fold_externo"]), "n": len(idx), "segundos_shap": segundos}


def shap_oof(tarea: str, filas: pd.DataFrame, X: pd.DataFrame, y: pd.Series, n_workers: int | None = None,
             lotes_por_fold: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valores SHAP de todos los vinos con el modelo de su fold externo.

    Parameters
    ----------
    filas : DataFrame
        Las 5 filas del log de la combinación a explicar (``filas_combinacion``).
    lotes_por_fold : int
        Lotes en que se divide la prueba de cada fold para calcularlos en paralelo.

    Returns
    -------
    (DataFrame, DataFrame)
        Valores SHAP (una fila por vino, una columna por variable, más ``base``)
        y tiempos de cálculo por lote.
    """
    from joblib import Parallel, delayed

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    trabajos = []
    for fila in filas.to_dict("records"):
        _, te = particiones_externas(tarea, X, y)[int(fila["fold_externo"])]
        for posiciones in np.array_split(np.arange(len(te)), lotes_por_fold):
            trabajos.append(delayed(_shap_lote)(tarea, fila, X, y, posiciones))
    resultados = Parallel(n_jobs=n_workers or config.N_WORKERS, batch_size=1)(trabajos)
    valores = pd.concat([r[0] for r in resultados]).sort_values("idx").reset_index(drop=True)
    return valores, pd.DataFrame([r[1] for r in resultados])


def importancia_shap(valores: pd.DataFrame) -> pd.DataFrame:
    """Media de |SHAP| por variable (global y por fold) y rango de cada variable."""
    global_ = valores[VARIABLES].abs().mean()
    por_fold = valores.groupby("fold")[VARIABLES].apply(lambda g: g.abs().mean())
    rangos = por_fold.rank(axis=1, ascending=False)
    tabla = pd.DataFrame({"media_abs_shap": global_, "rango": global_.rank(ascending=False).astype(int),
                          "rango_min_fold": rangos.min().astype(int), "rango_max_fold": rangos.max().astype(int),
                          "cv_entre_folds": por_fold.std() / por_fold.mean()})
    return tabla.sort_values("media_abs_shap", ascending=False).rename_axis("variable").reset_index()


def codificar(X: pd.DataFrame) -> np.ndarray:
    """Matriz numérica para LIME (densidad en g/L y tipo_vino como 0/1)."""
    Z = X[VARIABLES].copy()
    Z["density"] = Z["density"] * 1000
    Z["tipo_vino"] = (Z["tipo_vino"] == TIPOS[1]).astype(float)
    return Z.to_numpy(dtype=float)


def decodificar(Z: np.ndarray) -> pd.DataFrame:
    """Inversa de ``codificar``: vuelve a un DataFrame que el pipeline acepta."""
    X = pd.DataFrame(np.asarray(Z, float), columns=VARIABLES)
    X["density"] = X["density"] / 1000
    X["tipo_vino"] = np.where(X["tipo_vino"].round() >= 1, TIPOS[1], TIPOS[0])
    return X


def explicador_lime(tarea: str, X_entrenamiento: pd.DataFrame, semilla: int = config.RANDOM_STATE,
                    nombres=None):
    """``LimeTabularExplainer`` con los datos de entrenamiento del fold."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(codificar(X_entrenamiento), feature_names=nombres or VARIABLES,
                                categorical_features=[len(VARIABLES) - 1],
                                categorical_names={len(VARIABLES) - 1: ["tinto", "blanco"]},
                                mode="classification" if tarea == "clasificacion" else "regression",
                                discretize_continuous=True, random_state=semilla)


def explicar_lime(explicador, pipe, tarea: str, x: pd.DataFrame, num_samples: int = 5000) -> pd.DataFrame:
    """Pesos LIME de un vino (``x``: DataFrame de una fila).

    Returns
    -------
    DataFrame
        ``variable``, ``condicion``, ``peso``, intercepto, predicción local y R² local,
        ordenado por |peso|.
    """
    if tarea == "clasificacion":
        def predecir(Z):
            return pipe.predict_proba(decodificar(Z))
        exp = explicador.explain_instance(codificar(x)[0], predecir, labels=(1,),
                                          num_features=len(VARIABLES), num_samples=num_samples)
        etiqueta = 1
    else:
        def predecir(Z):
            return pipe.predict(decodificar(Z))
        exp = explicador.explain_instance(codificar(x)[0], predecir, num_features=len(VARIABLES),
                                          num_samples=num_samples)
        etiqueta = 0 if 0 in exp.as_map() else next(iter(exp.as_map()))
    condiciones = dict(zip([i for i, _ in exp.as_map()[etiqueta]], [c for c, _ in exp.as_list(label=etiqueta)]))
    filas = [{"variable": VARIABLES[i], "condicion": condiciones[i], "peso": w} for i, w in exp.as_map()[etiqueta]]
    salida = pd.DataFrame(filas)
    salida["intercepto"] = float(np.atleast_1d(exp.intercept[etiqueta])[0]) if isinstance(exp.intercept, dict) \
        else float(exp.intercept)
    salida["prediccion_local"] = float(np.atleast_1d(exp.local_pred)[0])
    salida["r2_local"] = float(exp.score)
    return salida.reindex(salida["peso"].abs().sort_values(ascending=False).index).reset_index(drop=True)
