from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .original_model import auc_mann_whitney


def holm(p):
    """p-valores ajustados por Holm-Bonferroni (en el orden original)."""
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    m = p.size
    ajust = np.maximum.accumulate((m - np.arange(m)) * p[orden])
    salida = np.empty(m)
    salida[orden] = np.minimum(ajust, 1.0)
    return salida


def wilcoxon_pareado(a, b):
    """Prueba de rangos con signo de Wilcoxon (bilateral) sobre diferencias pareadas a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    r = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    n = int((d != 0).sum())
    rangos = stats.rankdata(np.abs(d[d != 0]))
    r_mas, r_menos = rangos[d[d != 0] > 0].sum(), rangos[d[d != 0] < 0].sum()
    return {"estadistico": float(r.statistic), "p": float(r.pvalue), "n_no_nulas": n,
            "rank_biserial": float((r_mas - r_menos) / (r_mas + r_menos)) if n else 0.0,
            "positivas": int((d > 0).sum())}


def t_corregida(dif, n_entrenamiento, n_prueba, alfa=0.05):
    """t de Student corregida de Nadeau y Bengio para validación cruzada repetida (Bouckaert y Frank, 2004).

    La varianza de la media se infla por (1/J + n_prueba/n_entrenamiento) en lugar de 1/J.
    """
    d = np.asarray(dif, float)
    J = d.size
    var = d.var(ddof=1)
    ee = np.sqrt((1.0 / J + n_prueba / n_entrenamiento) * var)
    t = d.mean() / ee if ee > 0 else np.inf * np.sign(d.mean())
    gl = J - 1
    q = stats.t.ppf(1 - alfa / 2, gl)
    return {"estadistico": float(t), "gl": gl, "p": float(2 * stats.t.sf(abs(t), gl)), "media": float(d.mean()),
            "ic_inf": float(d.mean() - q * ee), "ic_sup": float(d.mean() + q * ee), "ee": float(ee)}


def _componentes_delong(y, puntajes):
    y = np.asarray(y).astype(bool)
    puntajes = np.atleast_2d(np.asarray(puntajes, dtype=float))
    pos, neg = puntajes[:, y], puntajes[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([stats.rankdata(f) for f in pos])
    ty = np.vstack([stats.rankdata(f) for f in neg])
    tz = np.vstack([stats.rankdata(f) for f in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n


def delong(y, s_a, s_b, alfa=0.05):
    """Prueba de DeLong para AUC correlacionados (algoritmo rápido de Sun y Xu, 2014)."""
    aucs, cov = _componentes_delong(y, np.vstack([s_a, s_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    dif = aucs[0] - aucs[1]
    ee = float(np.sqrt(max(var, 1e-300)))
    zc = stats.norm.ppf(1 - alfa / 2)
    z = dif / ee
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diferencia": float(dif), "ee": ee,
            "ic_inf": float(dif - zc * ee), "ic_sup": float(dif + zc * ee), "estadistico": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def bootstrap_bca_dif_auc(y, s_a, s_b, B=2000, alfa=0.05, semilla=0):
    """Intervalo BCa de AUC(a) - AUC(b) con remuestreo pareado y estratificado de los casos."""
    y = np.asarray(y).astype(int)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    rng = np.random.default_rng(semilla)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    est = auc_mann_whitney(y, s_a) - auc_mann_whitney(y, s_b)
    boot = np.empty(B)
    for i in range(B):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        boot[i] = auc_mann_whitney(y[idx], s_a[idx]) - auc_mann_whitney(y[idx], s_b[idx])
    z0 = stats.norm.ppf(np.clip((boot < est).mean() + 0.5 * (boot == est).mean(), 1e-6, 1 - 1e-6))
    grupos = np.array_split(np.random.default_rng(semilla + 1).permutation(y.size), 50)
    jack = np.array([auc_mann_whitney(np.delete(y, g), np.delete(s_a, g)) - auc_mann_whitney(np.delete(y, g), np.delete(s_b, g))
                     for g in grupos])
    dj = jack.mean() - jack
    acel = (dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5) if (dj ** 2).sum() > 0 else 0.0
    zs = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
    cuant = stats.norm.cdf(z0 + (z0 + zs) / (1 - acel * (z0 + zs)))
    lo, hi = np.quantile(boot, cuant)
    return {"diferencia": float(est), "ic_inf": float(lo), "ic_sup": float(hi), "z0": float(z0), "aceleracion": float(acel),
            "B": B}


def delta_cliff(a, b):
    """Delta de Cliff entre dos muestras y su magnitud según Romano et al. (2006)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float((np.sign(a[:, None] - b[None, :])).mean())
    ad = abs(d)
    mag = "despreciable" if ad < 0.147 else "pequeño" if ad < 0.33 else "mediano" if ad < 0.474 else "grande"
    return d, mag


def d_cohen_pareada(dif):
    """d de Cohen para diferencias pareadas: media / desviación estándar."""
    d = np.asarray(dif, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.inf


def friedman(tabla: pd.DataFrame, mayor_es_mejor=True, alfa=0.05):
    """Friedman con corrección de Iman-Davenport, rangos promedio, CD de Nemenyi y W de Kendall.

    Parameters
    ----------
    tabla : DataFrame bloques x modelos.
    """
    rangos = (-tabla if mayor_es_mejor else tabla).rank(axis=1, method="average")
    N, k = tabla.shape
    chi2, p = stats.friedmanchisquare(*[tabla[c] for c in tabla.columns])
    f_id = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = stats.f.sf(f_id, k - 1, (k - 1) * (N - 1))
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return {"rangos": rangos.mean().sort_values(), "N": N, "k": k, "chi2": float(chi2), "p": float(p),
            "F_ID": float(f_id), "p_ID": float(p_id), "cd": float(q * np.sqrt(k * (k + 1) / (6 * N))),
            "W": float(chi2 / (N * (k - 1)))}


def nemenyi_p(rangos: pd.Series, N: int) -> pd.DataFrame:
    """p-valores de Nemenyi para todos los pares a partir de los rangos promedio."""
    k = rangos.size
    ee = np.sqrt(k * (k + 1) / (6 * N))
    r = rangos.to_numpy()
    q = np.abs(r[:, None] - r[None, :]) / ee * np.sqrt(2)
    return pd.DataFrame(stats.studentized_range.sf(q, k, np.inf), index=rangos.index, columns=rangos.index)


def bayes_t_correlada(dif, rho, rope):
    """Prueba t bayesiana correlacionada (Corani y Benavoli, 2015) con región de equivalencia práctica."""
    d = np.asarray(dif, float)
    n = d.size
    escala = np.sqrt((1.0 / n + rho / (1.0 - rho)) * d.var(ddof=1))
    post = stats.t(df=n - 1, loc=d.mean(), scale=escala)
    p_izq, p_der = post.cdf(-rope), post.sf(rope)
    return {"p_peor": float(p_izq), "p_equivalente": float(1 - p_izq - p_der), "p_mejor": float(p_der)}


def bayes_rangos_signo(dif, rope, n_muestras=50000, prior=0.5, semilla=0):
    """Prueba bayesiana de rangos con signo (Benavoli et al., 2017) para diferencias entre conjuntos de datos."""
    d = np.concatenate([[0.0], np.asarray(dif, float)])
    rng = np.random.default_rng(semilla)
    pesos = rng.dirichlet(np.r_[prior, np.ones(d.size - 1)], n_muestras)
    sumas = d[:, None] + d[None, :]
    regiones = [(sumas < -2 * rope), (np.abs(sumas) <= 2 * rope), (sumas > 2 * rope)]
    probs = np.stack([np.einsum("si,ij,sj->s", pesos, r.astype(float), pesos) for r in regiones], axis=1)
    ganador = probs.argmax(axis=1)
    return {"p_peor": float((ganador == 0).mean()), "p_equivalente": float((ganador == 1).mean()),
            "p_mejor": float((ganador == 2).mean())}
