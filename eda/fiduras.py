# -*- coding: utf-8 -*-
"""Genera las figuras (PNG) del EDA de wine quality."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

COLOR_RED = "#7B1E32"
COLOR_WHITE = "#C9A94A"
COLOR_TEXT = "#2B2320"
COLOR_GRID = "#E4DFD6"
BG = "#FFFFFF"

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": BG, "figure.facecolor": BG,
    "axes.edgecolor": COLOR_GRID, "grid.color": COLOR_GRID,
    "text.color": COLOR_TEXT, "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT, "ytick.color": COLOR_TEXT,
    "font.family": "DejaVu Sans",
})
PALETTE = {"red": COLOR_RED, "white": COLOR_WHITE}
FIG_DIR = "/home/claude/eda_figs"

red = pd.read_csv("/home/claude/wine_quality/winequality-red.csv", sep=";")
white = pd.read_csv("/home/claude/wine_quality/winequality-white.csv", sep=";")
red["tipo_vino"] = "red"
white["tipo_vino"] = "white"
df = pd.concat([red, white], ignore_index=True)
features = [c for c in df.columns if c not in ("tipo_vino", "quality")]

nombres_es = {
    "fixed acidity": "acidez fija", "volatile acidity": "acidez volátil",
    "citric acid": "ácido cítrico", "residual sugar": "azúcar residual",
    "chlorides": "cloruros", "free sulfur dioxide": "SO2 libre",
    "total sulfur dioxide": "SO2 total", "density": "densidad",
    "pH": "pH", "sulphates": "sulfatos", "alcohol": "alcohol",
    "quality": "calidad",
}

def guardar(fig, nombre, dpi=118):
    fig.savefig(f"{FIG_DIR}/{nombre}.png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  guardado: {nombre}.png")

# ------------------------------------------------------------------
# 1. Histogramas de todas las variables numéricas
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(features + ["quality"]):
    ax = axes[i]
    for tipo, color in PALETTE.items():
        subset = df[df["tipo_vino"] == tipo][col]
        ax.hist(subset, bins=30, alpha=0.55, color=color, label=tipo, density=True)
    ax.set_title(nombres_es.get(col, col), fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)
axes[0].legend(fontsize=8, frameon=False)
for j in range(len(features) + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Distribución de cada variable — Tinto vs Blanco", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
guardar(fig, "01_histogramas")

# ------------------------------------------------------------------
# 2. Boxplots para detección visual de atípicos
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(features + ["quality"]):
    ax = axes[i]
    data_to_plot = [df[df["tipo_vino"] == "red"][col], df[df["tipo_vino"] == "white"][col]]
    bp = ax.boxplot(data_to_plot, labels=["red", "white"], patch_artist=True,
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                                      markeredgecolor=COLOR_TEXT, alpha=0.4))
    for patch, color in zip(bp["boxes"], [COLOR_RED, COLOR_WHITE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(nombres_es.get(col, col), fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
for j in range(len(features) + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Boxplots por variable — detección de valores atípicos", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
guardar(fig, "02_boxplots")

# ------------------------------------------------------------------
# 3. Mapa de calor de correlaciones
# ------------------------------------------------------------------
corr = df[features + ["quality"]].rename(columns=nombres_es).corr()
fig, ax = plt.subplots(figsize=(10, 8))
cmap = sns.diverging_palette(10, 45, s=70, l=45, as_cmap=True, center="light")
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
            square=True, linewidths=0.5, linecolor=BG,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
ax.set_title("Matriz de correlación entre variables", fontsize=14, fontweight="bold", pad=14)
fig.tight_layout()
guardar(fig, "03_correlacion")

# ------------------------------------------------------------------
# 4. Distribución de quality por tipo de vino
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ct = pd.crosstab(df["quality"], df["tipo_vino"])
ct = ct[["red", "white"]]
x = np.arange(len(ct.index))
width = 0.38
ax.bar(x - width/2, ct["red"], width, label="red", color=COLOR_RED, alpha=0.85)
ax.bar(x + width/2, ct["white"], width, label="white", color=COLOR_WHITE, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(ct.index)
ax.set_xlabel("Puntaje de calidad")
ax.set_ylabel("Cantidad de vinos")
ax.set_title("Distribución de la calidad por tipo de vino", fontsize=14, fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
guardar(fig, "04_calidad_por_tipo")

# ------------------------------------------------------------------
# 5. Variables más correlacionadas con quality (boxplot vs quality)
# ------------------------------------------------------------------
top_vars = ["alcohol", "density", "volatile acidity", "chlorides"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col in zip(axes, top_vars):
    sns.boxplot(data=df, x="quality", y=col, ax=ax, color="#B8925A",
                fliersize=2, linewidth=1)
    ax.set_title(nombres_es.get(col, col), fontsize=12, fontweight="bold")
    ax.set_xlabel("Calidad")
    ax.set_ylabel("")
fig.suptitle("Variables más correlacionadas con la calidad", fontsize=14, fontweight="bold", y=1.04)
fig.tight_layout()
guardar(fig, "05_vars_vs_calidad")

# ------------------------------------------------------------------
# 6. Barras de correlación con quality (todas las variables)
# ------------------------------------------------------------------
corr_q = df[features].corrwith(df["quality"]).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = [COLOR_RED if v < 0 else "#3E6B4F" for v in corr_q.values]
ax.barh([nombres_es.get(c, c) for c in corr_q.index], corr_q.values, color=colors, alpha=0.85)
ax.axvline(0, color=COLOR_TEXT, linewidth=0.8)
ax.set_xlabel("Correlación de Pearson con calidad")
ax.set_title("¿Qué variables se relacionan más con la calidad?", fontsize=13, fontweight="bold")
fig.tight_layout()
guardar(fig, "06_barras_correlacion")

# ------------------------------------------------------------------
# 7. Comparación tinto vs blanco — variables más diferenciadoras
# ------------------------------------------------------------------
diff_vars = ["total sulfur dioxide", "volatile acidity", "residual sugar", "chlorides"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, col in zip(axes, diff_vars):
    for tipo, color in PALETTE.items():
        sns.kdeplot(df[df["tipo_vino"] == tipo][col], ax=ax, color=color,
                    fill=True, alpha=0.4, label=tipo, linewidth=1.5)
    ax.set_title(nombres_es.get(col, col), fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Densidad")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Tinto vs Blanco: las variables más diferenciadoras", fontsize=14, fontweight="bold", y=1.04)
fig.tight_layout()
guardar(fig, "07_tinto_vs_blanco")

# ------------------------------------------------------------------
# 8. % de atípicos por variable (barra horizontal)
# ------------------------------------------------------------------
outlier_pct = {}
for col in features + ["quality"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_pct[col] = ((df[col] < low) | (df[col] > high)).mean() * 100
outlier_s = pd.Series(outlier_pct).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([nombres_es.get(c, c) for c in outlier_s.index], outlier_s.values, color="#8C5E3C", alpha=0.85)
ax.set_xlabel("% de valores atípicos (método IQR 1.5x)")
ax.set_title("Proporción de atípicos por variable", fontsize=13, fontweight="bold")
fig.tight_layout()
guardar(fig, "08_pct_atipicos")

print("\nTodas las figuras generadas correctamente.")