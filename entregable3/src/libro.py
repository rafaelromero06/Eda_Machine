from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import RUTA_BASE

AZUL, NARANJA, GRIS, TINTA, TINTA_2, REJILLA = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
RESULTADOS = RUTA_BASE / "experiments" / "resultados"
TABLAS = RESULTADOS / "tablas"
ORIGINAL = "Bosque Ordinal Focalizado"
CORTOS = {"Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM", "KNN": "KNN", "Regresión Logística (L1/L2)": "RL",
          "Decision Tree": "DT", "Naive Bayes": "NB", "LightGBM": "LGBM", "CatBoost": "CB", ORIGINAL: "BOF"}


def configurar():
    """Estilo común de las figuras del libro."""
    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TINTA_2, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "legend.frameon": False, "axes.prop_cycle": matplotlib.cycler(color=[AZUL, NARANJA, "#1baf7a", "#eda100"]),
    })
    pd.set_option("display.precision", 4)


def tabla(nombre):
    return pd.read_csv(TABLAS / nombre)


def resultado(nombre):
    return pd.read_csv(RESULTADOS / nombre)


def json_tabla(nombre):
    return json.loads((TABLAS / nombre).read_text(encoding="utf-8"))


def p_texto(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def mostrar(df, formatos=None, ocultar_indice=True):
    """DataFrame con formato para el HTML del libro."""
    estilo = df.style.format(formatos or {}, na_rep="—")
    if ocultar_indice:
        estilo = estilo.hide(axis="index")
    return estilo.set_table_attributes('class="dataframe tabla-libro"')


def barras_horizontales(ax, etiquetas, valores, resaltar=None, error=None, formato="{:.3f}", xlim=None):
    colores = [AZUL if e == resaltar else GRIS for e in etiquetas]
    y = np.arange(len(etiquetas))
    ax.barh(y, valores, xerr=error, color=colores, height=0.55, error_kw={"elinewidth": 1, "ecolor": TINTA_2, "capsize": 0})
    ax.set_yticks(y, etiquetas)
    for yi, v, e in zip(y, valores, error if error is not None else [0] * len(valores)):
        ax.text(v + (e if e is not None else 0) + 0.002, yi, formato.format(v), va="center", fontsize=9, color=TINTA_2)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    return ax


def diagrama_cd(ax, rangos, cd, resaltar=ORIGINAL):
    r = rangos.sort_values()
    y = np.arange(len(r))[::-1]
    ax.scatter(r.values, y, s=60, c=[AZUL if m == resaltar else GRIS for m in r.index], zorder=3, edgecolors="white", linewidths=1.5)
    for xi, yi, m in zip(r.values, y, r.index):
        ax.text(xi + 0.12, yi, CORTOS.get(m, m), va="center", fontsize=9, color=TINTA)
    mejor = r.iloc[0]
    ax.axvspan(mejor, mejor + cd, color=AZUL, alpha=0.08, lw=0)
    ax.plot([mejor, mejor + cd], [len(r) - 0.3] * 2, color=TINTA, lw=2)
    ax.text(mejor + cd / 2, len(r) - 0.05, f"CD = {cd:.2f}", ha="center", fontsize=9, color=TINTA_2)
    ax.set_yticks([])
    ax.set_xlabel("Rango medio (1 = mejor)")
    ax.set_ylim(-0.8, len(r) + 0.4)
    ax.grid(axis="y", visible=False)
    return ax


def cajas(ax, tab, resaltar=ORIGINAL):
    orden = tab.mean().sort_values().index
    datos = [tab[m].to_numpy() for m in orden]
    partes = ax.boxplot(datos, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": TINTA, "lw": 1.5})
    for caja, m in zip(partes["boxes"], orden):
        caja.set_facecolor(AZUL if m == resaltar else "#f0efec")
        caja.set_edgecolor(AZUL if m == resaltar else GRIS)
        caja.set_alpha(0.35 if m == resaltar else 1)
    rng = np.random.default_rng(0)
    for i, m in enumerate(orden, 1):
        ax.scatter(tab[m], i + rng.uniform(-0.12, 0.12, len(tab)), s=10, color=AZUL if m == resaltar else GRIS, zorder=3)
    ax.set_yticks(range(1, len(orden) + 1), [CORTOS.get(m, m) for m in orden])
    ax.grid(axis="y", visible=False)
    return ax
