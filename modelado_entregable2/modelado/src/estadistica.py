from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def friedman_nemenyi(df: pd.DataFrame, tratamiento: str, bloques: list, metrica: str,
                     mayor_es_mejor: bool = True, alfa: float = 0.05) -> dict:
    """Prueba de Friedman, corrección de Iman-Davenport, CD de Nemenyi y W de Kendall.

    Parameters
    ----------
    df : DataFrame en formato largo (una fila por tratamiento y bloque).
    tratamiento : columna con los tratamientos que se comparan (p. ej. ``modelo``).
    bloques : columnas que definen un bloque (p. ej. ``["balanceo", "optimizador", "fold_externo"]``).
    metrica : columna con la métrica.
    mayor_es_mejor : si True, el rango 1 es el valor más alto.

    Returns
    -------
    dict
        ``rangos`` (Serie ordenada), ``cd``, ``N``, ``k``, ``chi2``, ``p``, ``F_ID``, ``p_ID``,
        ``W`` (W de Kendall, tamaño de efecto de Friedman) y ``tabla`` (bloques × tratamientos).
    """
    ancha = df.pivot_table(index=bloques, columns=tratamiento, values=metrica).dropna()
    rangos = (-ancha if mayor_es_mejor else ancha).rank(axis=1, method="average")
    N, k = ancha.shape
    chi2, p = stats.friedmanchisquare(*[ancha[c] for c in ancha.columns])
    f_id = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = stats.f.sf(f_id, k - 1, (k - 1) * (N - 1))
    q_alfa = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    cd = q_alfa * np.sqrt(k * (k + 1) / (6 * N))
    return {"rangos": rangos.mean().sort_values(), "cd": cd, "N": N, "k": k, "chi2": chi2, "p": p,
            "F_ID": f_id, "p_ID": p_id, "W": kendall_w(chi2, N, k), "tabla": ancha}


def kendall_w(chi2: float, N: int, k: int) -> float:
    """W de Kendall: chi2_F / (N (k − 1))."""
    return float(chi2 / (N * (k - 1)))


def _componentes_delong(y_true: np.ndarray, puntajes: np.ndarray):
    """AUC y matriz de covarianza de DeLong para k clasificadores (algoritmo de Sun y Xu)."""
    y = np.asarray(y_true).astype(bool)
    puntajes = np.atleast_2d(np.asarray(puntajes, dtype=float))
    pos, neg = puntajes[:, y], puntajes[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([stats.rankdata(fila) for fila in pos])
    ty = np.vstack([stats.rankdata(fila) for fila in neg])
    tz = np.vstack([stats.rankdata(fila) for fila in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong(y_true, puntaje_a, puntaje_b, alfa: float = 0.05) -> dict:
    """Prueba de DeLong para la diferencia de AUC de dos modelos sobre los mismos casos.

    Parameters
    ----------
    y_true : array (0/1).
    puntaje_a, puntaje_b : puntajes continuos de cada modelo (probabilidad o función de decisión).

    Returns
    -------
    dict
        ``auc_a``, ``auc_b``, ``diferencia`` (a − b), su error estándar, intervalo de confianza
        (1 − alfa), estadístico ``z`` y ``p`` bilateral.
    """
    aucs, cov = _componentes_delong(y_true, np.vstack([puntaje_a, puntaje_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    dif = aucs[0] - aucs[1]
    ee = float(np.sqrt(max(var, 1e-300)))
    z = dif / ee
    zc = stats.norm.ppf(1 - alfa / 2)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diferencia": float(dif), "ee": ee,
            "ic_inf": float(dif - zc * ee), "ic_sup": float(dif + zc * ee), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def auc_con_ic(y_true, puntaje, alfa: float = 0.05) -> tuple[float, float, float]:
    """AUC con intervalo de confianza de DeLong (transformación logit, que respeta [0, 1])."""
    aucs, cov = _componentes_delong(y_true, np.asarray(puntaje)[None, :])
    auc, var = float(aucs[0]), float(cov[0, 0])
    zc = stats.norm.ppf(1 - alfa / 2)
    logit = np.log(auc / (1 - auc))
    ee_logit = np.sqrt(var) / (auc * (1 - auc))
    inf, sup = logit - zc * ee_logit, logit + zc * ee_logit
    return auc, float(1 / (1 + np.exp(-inf))), float(1 / (1 + np.exp(-sup)))


def diebold_mariano(error_a, error_b, potencia: int = 2, h: int = 1) -> dict:
    """Prueba de Diebold-Mariano (corrección de Harvey, Leybourne y Newbold) sobre la pérdida ``|e|^potencia``.

    Returns
    -------
    dict
        ``perdida_a``, ``perdida_b``, ``diferencia`` (a − b; negativa = a mejor),
        estadístico ``dm`` y ``p`` bilateral (t de Student con n − 1 g.l.).
    """
    ea, eb = np.asarray(error_a, float), np.asarray(error_b, float)
    d = np.abs(ea) ** potencia - np.abs(eb) ** potencia
    n = len(d)
    dbar = d.mean()
    autocov = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = autocov[0] + 2 * sum(autocov[1:])
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    estad = dm * hln
    return {"perdida_a": float(np.mean(np.abs(ea) ** potencia)), "perdida_b": float(np.mean(np.abs(eb) ** potencia)),
            "diferencia": float(dbar), "dm": float(estad), "p": float(2 * stats.t.sf(abs(estad), df=n - 1))}


def conjunto_confianza_modelos(perdidas: pd.DataFrame, alfa: float = 0.10, B: int = 2000,
                               semilla: int = 42) -> pd.DataFrame:
    """Model Confidence Set (MCS) con el estadístico T_max y bootstrap i.i.d.

    Parameters
    ----------
    perdidas : DataFrame (casos × modelos)
        Pérdida de cada modelo en cada caso (p. ej. error cuadrático o puntaje de Brier).
    alfa : nivel del conjunto; el MCS contiene los modelos con p_MCS ≥ alfa.
    B : réplicas bootstrap.

    Returns
    -------
    DataFrame
        Una fila por modelo con su pérdida media, el paso en que se eliminó, el
        p-valor MCS y si pertenece al conjunto, ordenado del mejor al peor.
    """
    L = perdidas.to_numpy(dtype=float)
    n, m = L.shape
    rng = np.random.default_rng(semilla)
    medias = L.mean(axis=0)
    medias_b = np.empty((B, m))
    for ini in range(0, B, 250):
        fin = min(B, ini + 250)
        conteos = np.vstack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(fin - ini)])
        medias_b[ini:fin] = conteos @ L / n

    vivos = list(range(m))
    p_mcs = np.ones(m)
    paso = np.zeros(m, dtype=int)
    p_acumulado, k = 0.0, 0
    while len(vivos) > 1:
        k += 1
        mu, mub = medias[vivos], medias_b[:, vivos]
        d = mu - mu.mean()
        db = mub - mub.mean(axis=1, keepdims=True)
        ee = np.sqrt(((db - d) ** 2).mean(axis=0))
        t = d / ee
        t_max_b = ((db - d) / ee).max(axis=1)
        p = float((t_max_b >= t.max()).mean())
        p_acumulado = max(p_acumulado, p)
        peor = vivos[int(np.argmax(t))]
        p_mcs[peor], paso[peor] = p_acumulado, k
        vivos.remove(peor)
    paso[vivos[0]] = k + 1
    salida = pd.DataFrame({"modelo": perdidas.columns, "perdida_media": medias, "paso_eliminacion": paso,
                           "p_mcs": p_mcs})
    salida["en_mcs"] = salida["p_mcs"] >= alfa
    return salida.sort_values(["p_mcs", "perdida_media"], ascending=[False, True]).reset_index(drop=True)


def magnitud_delta(delta: float) -> str:
    """Magnitud de un delta tipo Cliff (|δ|) con los umbrales de Romano et al. (2006)."""
    a = abs(delta)
    if a < 0.147:
        return "insignificante"
    if a < 0.33:
        return "pequeño"
    if a < 0.474:
        return "mediano"
    return "grande"


def efecto_pareado(a, b, B: int = 5000, semilla: int = 42, alfa: float = 0.05) -> dict:
    """Tamaño de efecto de la diferencia pareada a − b (mismos bloques).

    Returns
    -------
    dict
        ``diferencia`` media con su intervalo bootstrap, ``rank_biserial``, ``prob_superioridad``,
        ``d_z``, ``magnitud`` y ``p_wilcoxon``.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    no_nulos = d[d != 0]
    if len(no_nulos):
        r = stats.rankdata(np.abs(no_nulos))
        rb = (r[no_nulos > 0].sum() - r[no_nulos < 0].sum()) / r.sum()
        p_w = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue) if len(no_nulos) >= 2 else np.nan
    else:
        rb, p_w = 0.0, 1.0
    sd = d.std(ddof=1)
    return {"diferencia": float(d.mean()), "ic_inf": float(np.quantile(medias, alfa / 2)),
            "ic_sup": float(np.quantile(medias, 1 - alfa / 2)), "rank_biserial": float(rb),
            "prob_superioridad": float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / len(d)),
            "d_z": float(d.mean() / sd) if sd > 0 else np.nan, "magnitud": magnitud_delta(rb), "p_wilcoxon": p_w}


def holm(pvalores) -> np.ndarray:
    """Corrección de Holm-Bonferroni (controla la tasa de error por familia)."""
    p = np.asarray(pvalores, float)
    orden = np.argsort(p)
    ajustados = np.empty_like(p)
    m = len(p)
    acumulado = 0.0
    for i, j in enumerate(orden):
        acumulado = max(acumulado, (m - i) * p[j])
        ajustados[j] = min(1.0, acumulado)
    return ajustados


def ic_bootstrap(metrica, y_true, y_pred, B: int = 2000, semilla: int = 42, alfa: float = 0.05,
                 estratificar: bool = False) -> tuple[float, float, float]:
    """Valor de una métrica e intervalo de confianza bootstrap (percentil).

    Parameters
    ----------
    metrica : callable(y_true, y_pred) -> float
    estratificar : si True, remuestrea dentro de cada clase.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    grupos = [np.flatnonzero(y_true == c) for c in np.unique(y_true)] if estratificar else [np.arange(n)]
    valores = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in grupos])
        valores[b] = metrica(y_true[idx], y_pred[idx])
    return float(metrica(y_true, y_pred)), float(np.quantile(valores, alfa / 2)), float(np.quantile(valores, 1 - alfa / 2))
