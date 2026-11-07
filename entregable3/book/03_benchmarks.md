---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Modelos benchmark y mejores combinaciones

```{code-cell} ipython3
:tags: [remove-cell]
import sys, json, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from src.libro import configurar, mostrar, tabla, resultado, barras_horizontales, AZUL, GRIS, ORIGINAL, CORTOS
from src.benchmarks import BENCHMARKS_CLASE, BENCHMARKS_SOTA, ESPACIOS, MEJORES
configurar()
E2 = RAIZ.parent / "modelado_entregable2" / "modelado" / "experiments"
```

## Criterio de selección

Del universo de 136 combinaciones del Entregable 2 (140 menos las 4 de KNN con `class_weight`, que no existen porque KNN no admite pesos por clase), se elige para cada uno de los 7 modelos base la combinación de balanceo y optimizador con el **mayor AUC medio del bucle externo** de la validación anidada 5 × 3:

```{code-cell} ipython3
:tags: [hide-input]
log = pd.read_csv(E2 / "logs" / "resultados_clasificacion_p40.csv")
g = log.groupby(["modelo", "balanceo", "optimizador"])["auc"].agg(["mean", "std"]).reset_index()
filas = []
for m in BENCHMARKS_CLASE:
    b, o = MEJORES[m]
    f = g[(g["modelo"] == m) & (g["balanceo"] == b) & (g["optimizador"] == o)].iloc[0]
    alternativas = len(g[g["modelo"] == m])
    filas.append({"Modelo": m, "Balanceo": b, "Optimizador": o, "AUC externo (media ± DE)": f"{f['mean']:.4f} ± {f['std']:.4f}",
                  "Combinaciones evaluadas": alternativas,
                  "Motivo": "mayor AUC medio externo entre sus combinaciones" + (" (empate exacto con sin balanceo)" if m == "Naive Bayes" else "")})
sel = pd.DataFrame(filas)
mostrar(sel)
```

La **mejor combinación global** es Random Forest con `class_weight` y Grid Search (AUC 0.8605 ± 0.0152). En la comparación de Friedman y Nemenyi del Entregable 2 (60 bloques: pliegue × balanceo × optimizador), Random Forest tuvo el mejor rango medio (1.33) y formó el grupo de finalistas junto con XGBoost (1.72): su diferencia de rangos (0.39) es menor que la diferencia crítica (1.16). En el *Model Confidence Set* del Entregable 2, las configuraciones de Random Forest con `class_weight` también quedaron en el conjunto de los mejores. La selección es, por lo tanto, coherente con la comparación estadística y no solo con la métrica puntual.

## Métodos recientes del estado del arte

Además de los 7 modelos vistos en clase se agregan dos métodos de *gradient boosting* posteriores a XGBoost, que son la referencia en datos tabulares {cite:p}`grinsztajn2022`:

- **LightGBM** {cite:p}`ke2017`: árboles que crecen por hojas con histogramas y muestreo de gradientes.
- **CatBoost** {cite:p}`prokhorenkova2018`: *boosting* ordenado con árboles simétricos, que reduce el sesgo de predicción del *boosting* clásico.

Ambos se optimizan con el mismo método bayesiano (TPE) y presupuesto que el modelo original. La ponderación de clases es un hiperparámetro más (balanceo dentro de la búsqueda) y la primera evaluación usa los hiperparámetros por defecto de cada librería, como pide el enunciado.

```{code-cell} ipython3
:tags: [hide-input]
def rango(spec):
    if spec["type"] == "categorical":
        return ", ".join(str(c) for c in spec["categories"])
    return f"{spec['low']:g} – {spec['high']:g}" + (" (log)" if spec["log"] else "")
filas = []
for m in BENCHMARKS_CLASE + BENCHMARKS_SOTA + [ORIGINAL]:
    for h, spec in ESPACIOS[m]["space"].items():
        filas.append({"Modelo": m, "Hiperparámetro": h, "Rango": rango(spec)})
espacios = pd.DataFrame(filas)
dims = espacios.groupby("Modelo", sort=False).size()
glue("dims_bof", int(dims[ORIGINAL]), display=False)
mostrar(espacios)
```

## Equidad de la comparación

Todos los modelos usan las mismas particiones externas e internas, el mismo preprocesamiento (escalado y one-hot dentro del pliegue), la misma métrica interna (AUC) y **40 evaluaciones** de hiperparámetros por pliegue externo; Grid Search usa una grilla adaptativa de a lo sumo 40 puntos. El espacio del modelo original tiene {glue:text}`dims_bof` dimensiones, más que Random Forest (4) y XGBoost (5) y una menos que LightGBM (7): con el mismo presupuesto, buscar en un espacio más grande es una desventaja para el modelo original, no una ventaja. Al modelo original no se le encoló ninguna configuración inicial.

```{code-cell} ipython3
:tags: [hide-input]
cv = resultado("cv_filas.csv")
v = cv[cv["conjunto"] == "vino"].groupby("modelo").agg(evaluaciones=("n_evaluaciones", "mean"), busqueda=("tiempo_busqueda_s", "mean"),
                                                       ajuste=("tiempo_ajuste_s", "mean"), prediccion=("tiempo_prediccion_ms_por_caso", "mean"))
v["horas_cpu"] = cv[cv["conjunto"] == "vino"].groupby("modelo")["tiempo_busqueda_s"].sum() / 3600
mem = resultado("memoria_modelos.csv").set_index("modelo")
v = v.join(mem[["memoria_ajuste_mb", "tamano_en_memoria_mb"]])
v = v.sort_values("busqueda", ascending=False)
v.index.name = "Modelo"
mostrar(v.reset_index().rename(columns={"evaluaciones": "Evaluaciones por pliegue", "busqueda": "Búsqueda por pliegue (s)", "ajuste": "Ajuste final (s)",
                                        "prediccion": "Predicción (ms/vino)", "horas_cpu": "Horas de CPU (25 pliegues)",
                                        "memoria_ajuste_mb": "Memoria del ajuste (MB)", "tamano_en_memoria_mb": "Tamaño del modelo (MB)"}),
        {"Evaluaciones por pliegue": "{:.0f}", "Búsqueda por pliegue (s)": "{:.0f}", "Ajuste final (s)": "{:.2f}", "Predicción (ms/vino)": "{:.3f}",
         "Horas de CPU (25 pliegues)": "{:.1f}", "Memoria del ajuste (MB)": "{:.0f}", "Tamaño del modelo (MB)": "{:.1f}"})
```

Presupuesto realmente consumido por modelo en vino (25 pliegues). El tiempo se midió con 11 procesos en paralelo en un AMD Ryzen 5 5600GT (6 núcleos, 15.8 GB); la memoria se midió aparte, ajustando cada modelo final en un proceso nuevo.

## Resultados de las mejores combinaciones

```{code-cell} ipython3
:tags: [hide-input]
res = tabla("resumen_modelos.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
r = res.sort_values("auc_media")
barras_horizontales(ax, r["modelo"].tolist(), r["auc_media"].to_numpy(), resaltar=ORIGINAL, error=r["auc_sd"].to_numpy(), formato="{:.4f}",
                    xlim=(0.70, 0.92))
ax.set_xlabel("AUC-ROC medio en 25 particiones (± 1 DE)")
ax.set_title("AUC de la mejor combinación de cada modelo")
plt.tight_layout()
glue("fig_resumen", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_resumen
:name: fig-resumen
AUC medio de las 25 particiones (5 repeticiones × 5 pliegues) de cada modelo. En azul, el modelo original.
```

```{code-cell} ipython3
:tags: [hide-input]
t = res[["modelo", "balanceo", "optimizador", "auc_media", "auc_sd", "auc_oof_rep1", "auc_ic_inf", "auc_ic_sup", "pr_auc_media", "brier_media", "f1_media"]].copy()
t["IC 95 % (repetición 1)"] = t.apply(lambda f: f"[{f.auc_ic_inf:.3f}, {f.auc_ic_sup:.3f}]", axis=1)
mostrar(t.drop(columns=["auc_ic_inf", "auc_ic_sup"]).rename(columns={"modelo": "Modelo", "balanceo": "Balanceo", "optimizador": "Optimizador",
        "auc_media": "AUC media", "auc_sd": "DE", "auc_oof_rep1": "AUC repetición 1", "pr_auc_media": "PR-AUC", "brier_media": "Brier", "f1_media": "F1 (0.5)"}),
        {"AUC media": "{:.4f}", "DE": "{:.4f}", "AUC repetición 1": "{:.4f}", "PR-AUC": "{:.4f}", "Brier": "{:.4f}", "F1 (0.5)": "{:.3f}"})
```

PENDIENTE: interpretación de la tabla de resultados.

## Reentrenamiento final y serialización

Con los hiperparámetros que eligió una última búsqueda sobre los 5320 vinos (mismo optimizador, presupuesto y validación interna de 3 pliegues), cada pipeline completo (preprocesamiento, balanceo y modelo) se reentrenó con todos los datos y se guardó en `artifacts/<modelo>/modelo.joblib` con `joblib.dump(..., compress=3)`. Junto a cada modelo hay un `metadata.json` (versiones de las librerías, semilla, hiperparámetros, esquema de las columnas de entrada con sus rangos válidos, métricas del bucle externo, tamaño y huella SHA-256) y las predicciones de 200 vinos fijos que verifican la ida y vuelta.

```{code-cell} ipython3
:tags: [hide-input]
filas = []
for carpeta in sorted((RAIZ / "artifacts").iterdir()):
    meta = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    red = meta.get("reduccion_arboles")
    filas.append({"Modelo": meta["modelo"], "Hiperparámetros finales": ", ".join(f"{k}={v}" for k, v in meta["hiperparametros_finales"].items()),
                  "Tamaño (MB)": meta["tamano_archivo_mb"], "Ida y vuelta idéntica": "sí" if meta["prueba_ida_y_vuelta"]["identicas"] else "no",
                  "Diferencia máxima": meta["prueba_ida_y_vuelta"]["diferencia_maxima"],
                  "Árboles desplegados": f"{red['arboles_desplegados']} de {red['arboles_busqueda']}" if red else "—"})
artef = pd.DataFrame(filas)
mostrar(artef, {"Tamaño (MB)": "{:.2f}", "Diferencia máxima": "{:.1e}"})
```

PENDIENTE: comentario sobre tamaños y reducción de árboles.
