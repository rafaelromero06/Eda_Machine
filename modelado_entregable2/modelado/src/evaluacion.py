from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import bds

_EPS = 1e-6


def curva_confiabilidad(y_true, prob, n_bins: int = 10, estrategia: str = "quantile") -> pd.DataFrame:
    """Probabilidad media predicha frente a frecuencia observada por intervalo.

    Parameters
    ----------
    estrategia : {"quantile", "uniform"}
        Intervalos con el mismo número de casos o del mismo ancho.
    """
    y, p = np.asarray(y_true, float), np.asarray(prob, float)
    if estrategia == "quantile":
        bordes = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        bordes = np.linspace(0, 1, n_bins + 1)
    grupo = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, len(bordes) - 2)
    tabla = (pd.DataFrame({"grupo": grupo, "p": p, "y": y}).groupby("grupo")
             .agg(prob_media=("p", "mean"), frecuencia=("y", "mean"), n=("y", "size")).reset_index(drop=True))
    return tabla


def ece(y_true, prob, n_bins: int = 10, estrategia: str = "uniform") -> float:
    """Error de calibración esperado: promedio ponderado de |frecuencia − probabilidad|."""
    t = curva_confiabilidad(y_true, prob, n_bins, estrategia)
    return float((t["n"] * (t["frecuencia"] - t["prob_media"]).abs()).sum() / t["n"].sum())


def calibracion_logistica(y_true, prob) -> tuple[float, float]:
    """Intercepto a y pendiente b de logit(P(y=1)) = a + b · logit(p)."""
    p = np.clip(np.asarray(prob, float), _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p))
    ajuste = sm.Logit(np.asarray(y_true, float), sm.add_constant(logit)).fit(disp=0)
    return float(ajuste.params[0]), float(ajuste.params[1])


def metricas_probabilisticas(y_true, prob) -> dict:
    """Métricas de discriminación y calibración de un vector de probabilidades."""
    y, p = np.asarray(y_true, float), np.clip(np.asarray(prob, float), 0, 1)
    prevalencia = y.mean()
    brier = brier_score_loss(y, p)
    referencia = prevalencia * (1 - prevalencia)
    intercepto, pendiente = calibracion_logistica(y, p)
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier,
            "brier_skill": 1 - brier / referencia, "log_loss": log_loss(y, np.clip(p, _EPS, 1 - _EPS)),
            "ece": ece(y, p), "intercepto_cal": intercepto, "pendiente_cal": pendiente,
            "prob_media": p.mean(), "prevalencia": prevalencia}


def pruebas_residuos(residuos, exog: pd.DataFrame, lags=(10, 20), bds_dim: int = 3) -> pd.DataFrame:
    """Batería de pruebas sobre los residuos de un modelo de regresión.

    Parameters
    ----------
    residuos : array (observado − predicho), en el orden original de los datos.
    exog : DataFrame con las variables explicativas numéricas (para Breusch-Pagan y White).
    lags : rezagos de Ljung-Box.
    bds_dim : dimensión máxima de la prueba BDS.

    Returns
    -------
    DataFrame
        Prueba, hipótesis nula, estadístico y p-valor.
    """
    e = np.asarray(residuos, float)
    Z = sm.add_constant(np.asarray(exog, float))
    filas = []
    jb, p_jb, asimetria, curtosis = jarque_bera(e)
    filas.append(("Jarque-Bera", "Residuos normales", jb, p_jb,
                  f"asimetría {asimetria:.2f}, curtosis {curtosis:.2f}"))
    lm, p_lm, _, _ = het_breuschpagan(e, Z)
    filas.append(("Breusch-Pagan", "Varianza constante (lineal en las variables)", lm, p_lm, ""))
    lm, p_lm, _, _ = het_white(e, Z)
    filas.append(("White", "Varianza constante (cuadrados y productos cruzados)", lm, p_lm, ""))
    lb = acorr_ljungbox(e, lags=list(lags), return_df=True)
    for lag, fila in lb.iterrows():
        filas.append((f"Ljung-Box (rezago {lag})", "Sin autocorrelación en el orden de los datos",
                      fila["lb_stat"], fila["lb_pvalue"], ""))
    filas.append(("Durbin-Watson", "Sin autocorrelación de orden 1 (≈ 2)", durbin_watson(e), np.nan, ""))
    est_bds, p_bds = bds((e - e.mean()) / e.std(), max_dim=bds_dim)
    for dim, (s, p) in enumerate(zip(np.atleast_1d(est_bds), np.atleast_1d(p_bds)), start=2):
        filas.append((f"BDS (dimensión {dim})", "Residuos i.i.d. (sin dependencia no lineal)", s, p, ""))
    return pd.DataFrame(filas, columns=["prueba", "hipotesis_nula", "estadistico", "p_valor", "detalle"])
