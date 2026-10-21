from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CLAVES = ["modelo", "balanceo", "optimizador"]
METRICAS = {"clasificacion": ["auc", "f1", "recall", "precision", "accuracy"],
            "regresion": ["rmse", "mae"]}
PRINCIPAL = {"clasificacion": ("auc", True), "regresion": ("rmse", False)}


def cargar_log(tarea: str) -> pd.DataFrame:
    """Log de la corrida final (presupuesto 40) de la tarea indicada."""
    ruta = config.RUTA_LOG_CLASIFICACION if tarea == "clasificacion" else config.RUTA_LOG_REGRESION
    return pd.read_csv(ruta)


def resumir(log: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Media y desviación estándar entre folds externos de cada combinación.

    Returns
    -------
    DataFrame
        Una fila por combinación con ``<métrica>``, ``<métrica>_std``, ``score_interno``,
        ``n_evaluaciones`` y ``tiempo_fold_s`` (medias de los 5 folds).
    """
    metricas = METRICAS[tarea]
    g = log.groupby(CLAVES)
    resumen = g[metricas].mean().join(g[metricas].std().add_suffix("_std"))
    resumen["score_interno"] = g["score_interno"].mean()
    resumen["n_evaluaciones"] = g["n_evaluaciones"].mean()
    resumen["tiempo_fold_s"] = g["tiempo_seg"].mean()
    return resumen.reset_index()


def mejor_por_modelo(resumen: pd.DataFrame) -> pd.DataFrame:
    """Combinación con mejor score interno medio de cada modelo."""
    return (resumen.sort_values("score_interno", ascending=False)
            .groupby("modelo", sort=False).head(1).reset_index(drop=True))


def candidatas_1ee(resumen: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Combinaciones a menos de un error estándar (d.e./√K) de la mejor métrica principal."""
    metrica, mayor = PRINCIPAL[tarea]
    mejor = resumen.loc[resumen[metrica].idxmax() if mayor else resumen[metrica].idxmin()]
    ee = mejor[f"{metrica}_std"] / np.sqrt(config.K_OUTER)
    if mayor:
        return resumen[resumen[metrica] >= mejor[metrica] - ee].copy()
    return resumen[resumen[metrica] <= mejor[metrica] + ee].copy()


def recomendada(resumen: pd.DataFrame, tarea: str) -> pd.Series:
    """Configuración  regla de 1 error estándar (desempate: F1 o tiempo)."""
    candidatas = candidatas_1ee(resumen, tarea)
    if tarea == "clasificacion":
        return candidatas.sort_values("f1", ascending=False).iloc[0]
    return candidatas.sort_values("tiempo_fold_s").iloc[0]


def etiqueta(fila, tarea: str) -> str:
    """Nombre corto de una combinación para tablas y gráficas."""
    corto = {"Regresión Logística (L1/L2)": "Reg. logística", "sin_balanceo": "sin balanceo"}
    modelo = corto.get(fila["modelo"], fila["modelo"])
    if tarea == "regresion":
        return f"{modelo} · {fila['optimizador']}"
    return f"{modelo} · {corto.get(fila['balanceo'], fila['balanceo'])} · {fila['optimizador']}"
