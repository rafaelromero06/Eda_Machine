# -*- coding: utf-8 -*-
"""
Análisis Exploratorio de Datos (EDA) — Wine Quality (red + white)
Genera: tablas (CSV) + figuras (PNG) que luego se ensamblan en un reporte HTML.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ------------------------------------------------------------------
# 0. Estilo visual (paleta temática: bordeaux para tinto, dorado para blanco)
# ------------------------------------------------------------------
COLOR_RED = "#7B1E32"     # bordeaux
COLOR_WHITE = "#C9A94A"   # dorado/paja
COLOR_TEXT = "#2B2320"
COLOR_GRID = "#E4DFD6"
BG = "#FFFFFF"

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": COLOR_GRID,
    "grid.color": COLOR_GRID,
    "text.color": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "font.family": "DejaVu Sans",
})
PALETTE = {"red": COLOR_RED, "white": COLOR_WHITE}

FIG_DIR = r"c:\Users\USER\OneDrive\Escritorio\eda\eda_figs"
OUT_DIR = r"c:\Users\USER\OneDrive\Escritorio\eda\outputs"

# ------------------------------------------------------------------
# 1. Cargar y combinar datos
# ------------------------------------------------------------------
red = pd.read_csv(r"C:\Users\USER\OneDrive\Escritorio\eda\winequality-red.csv", sep=";")
white = pd.read_csv(r"C:\Users\USER\OneDrive\Escritorio\eda\winequality-white.csv", sep=";")
red["tipo_vino"] = "red"
white["tipo_vino"] = "white"
df = pd.concat([red, white], ignore_index=True)

features = [c for c in df.columns if c not in ("tipo_vino", "quality")]

print("Shape:", df.shape)

# ------------------------------------------------------------------
# 2. Datos faltantes
# ------------------------------------------------------------------
missing = pd.DataFrame({
    "n_faltantes": df.isna().sum(),
    "pct_faltantes": (df.isna().sum() / len(df) * 100).round(2),
})
missing.to_csv(f"{OUT_DIR}/eda_01_datos_faltantes.csv")
print("\n=== Datos faltantes ===")
print(missing)

# ------------------------------------------------------------------
# 3. Duplicados
# ------------------------------------------------------------------
dup_total = df.duplicated().sum()
dup_red = red.duplicated().sum()
dup_white = white.duplicated().sum()
dup_summary = pd.DataFrame({
    "subset": ["red", "white", "combinado"],
    "n_filas": [len(red), len(white), len(df)],
    "n_duplicados": [dup_red, dup_white, dup_total],
    "pct_duplicados": [
        round(dup_red / len(red) * 100, 2),
        round(dup_white / len(white) * 100, 2),
        round(dup_total / len(df) * 100, 2),
    ],
})
dup_summary.to_csv(f"{OUT_DIR}/eda_02_duplicados.csv", index=False)
print("\n=== Duplicados ===")
print(dup_summary)

# ------------------------------------------------------------------
# 4. Estadísticas descriptivas
# ------------------------------------------------------------------
desc_overall = df[features + ["quality"]].describe().T
desc_overall.to_csv(f"{OUT_DIR}/eda_03_estadisticas_descriptivas.csv")
print("\n=== Estadísticas descriptivas (general) ===")
print(desc_overall.round(3))

desc_by_type = df.groupby("tipo_vino")[features + ["quality"]].mean().T
desc_by_type.columns = ["media_red", "media_white"]
desc_by_type.to_csv(f"{OUT_DIR}/eda_04_medias_por_tipo.csv")
print("\n=== Medias por tipo ===")
print(desc_by_type.round(3))

# ------------------------------------------------------------------
# 5. Detección de atípicos (IQR)
# ------------------------------------------------------------------
outlier_rows = []
for col in features + ["quality"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    outlier_rows.append({
        "variable": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "limite_inferior": low, "limite_superior": high,
        "n_atipicos": n_out, "pct_atipicos": round(n_out / len(df) * 100, 2),
    })
outliers_df = pd.DataFrame(outlier_rows).sort_values("pct_atipicos", ascending=False)
outliers_df.to_csv(f"{OUT_DIR}/eda_05_atipicos_iqr.csv", index=False)
print("\n=== Atípicos (método IQR, 1.5x) ===")
print(outliers_df.round(3).to_string(index=False))

# Valores extremos puntuales dignos de mención
top_extremos = pd.concat([
    df.nlargest(3, "residual sugar")[["tipo_vino", "residual sugar"]].assign(variable="residual sugar").rename(columns={"residual sugar": "valor"}),
    df.nlargest(3, "free sulfur dioxide")[["tipo_vino", "free sulfur dioxide"]].assign(variable="free sulfur dioxide").rename(columns={"free sulfur dioxide": "valor"}),
    df.nlargest(3, "density")[["tipo_vino", "density"]].assign(variable="density").rename(columns={"density": "valor"}),
])
top_extremos.to_csv(f"{OUT_DIR}/eda_06_valores_extremos_top.csv", index=False)
print("\n=== Top valores extremos individuales ===")
print(top_extremos)

# ------------------------------------------------------------------
# 6. Correlaciones
# ------------------------------------------------------------------
corr = df[features + ["quality"]].corr()
corr_quality = corr["quality"].drop("quality").sort_values(key=abs, ascending=False)
corr_quality.to_csv(f"{OUT_DIR}/eda_07_correlacion_con_quality.csv", header=["correlacion"])
print("\n=== Correlación con quality (ordenada por |valor|) ===")
print(corr_quality.round(3))

print("\nAnálisis numérico completo. Generando figuras...")