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

# Datos: ETL, EDA y auditoría de fuga

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from src.libro import configurar, mostrar, resultado, AZUL, NARANJA, GRIS, TINTA_2, CORTOS, ORIGINAL
from src.data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos, resumen_conjuntos
from src.original_model import auc_mann_whitney
configurar()
df = cargar_vinos()
bueno = (df["quality"] >= 7)
```

## ETL

Los archivos `winequality-red.csv` (1599 tintos) y `winequality-white.csv` (4898 blancos) se unen con una variable `tipo_vino`. De las 6497 filas, 1177 son duplicados exactos (misma química, mismo tipo y misma nota); se eliminan **antes** de partir los datos, porque un duplicado en entrenamiento y en prueba mide memoria y no generalización. Quedan 5320 vinos sin valores faltantes. La variable objetivo es el evento `quality >= 7`.

```{code-cell} ipython3
:tags: [hide-input]
prev = df.groupby("tipo_vino")["quality"].apply(lambda q: (q >= 7).mean())
glue("prev_total", f"{100 * bueno.mean():.1f}", display=False)
glue("prev_tinto", f"{100 * prev['red']:.1f}", display=False)
glue("prev_blanco", f"{100 * prev['white']:.1f}", display=False)
fig, ejes = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [2, 1]})
conteo = df["quality"].value_counts().sort_index()
ejes[0].bar(conteo.index.astype(str), conteo.values, color=[AZUL if q >= 7 else NARANJA for q in conteo.index], width=0.6)
ejes[0].set_xlabel("Nota de calidad")
ejes[0].set_ylabel("Número de vinos")
ejes[0].set_title("La nota se concentra en 5 y 6")
for x, v in zip(range(len(conteo)), conteo.values):
    ejes[0].text(x, v + 30, str(v), ha="center", fontsize=8, color=TINTA_2)
ejes[1].bar(["Tinto", "Blanco"], [100 * prev["red"], 100 * prev["white"]], color=AZUL, width=0.5)
ejes[1].set_ylabel("% de vinos buenos")
ejes[1].set_title("Prevalencia por tipo")
for x, v in enumerate([100 * prev["red"], 100 * prev["white"]]):
    ejes[1].text(x, v + 0.5, f"{v:.1f} %", ha="center", fontsize=9, color=TINTA_2)
ejes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AZUL), plt.Rectangle((0, 0), 1, 1, color=NARANJA)],
               labels=["Bueno (≥ 7)", "No bueno (≤ 6)"], loc="upper right")
plt.tight_layout()
glue("fig_objetivo", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_objetivo
:name: fig-objetivo
Distribución de la nota y prevalencia del evento por tipo de vino (5320 vinos sin duplicados).
```

El {glue:text}`prev_total` % de los vinos es bueno ({numref}`fig-objetivo`): la clase de interés es minoritaria, lo que justifica estudiar el balanceo y medir con AUC y PR-AUC además de exactitud. Los blancos tienen más vinos buenos ({glue:text}`prev_blanco` %) que los tintos ({glue:text}`prev_tinto` %), por lo que el tipo entra como variable.

## EDA

```{code-cell} ipython3
:tags: [hide-input]
aucs = {c: auc_mann_whitney(bueno.to_numpy(), df[c].to_numpy()) for c in NUMERICAS_VINO}
separacion = pd.Series({c: max(a, 1 - a) for c, a in aucs.items()}).sort_values(ascending=False)
top = list(separacion.index[:4])
fig, ejes = plt.subplots(2, 2, figsize=(9, 5.6))
for ax, c in zip(ejes.ravel(), top):
    lo, hi = df[c].quantile([0.005, 0.995])
    bins = np.linspace(lo, hi, 40)
    ax.hist(df.loc[~bueno, c], bins=bins, color=NARANJA, alpha=0.6, density=True, label="No bueno")
    ax.hist(df.loc[bueno, c], bins=bins, color=AZUL, alpha=0.6, density=True, label="Bueno")
    nombre, unidad, _ = DICCIONARIO_VINO[c]
    ax.set_xlabel(f"{nombre} ({unidad})")
    ax.set_ylabel("Densidad")
    ax.set_title(f"{nombre}: AUC de la variable sola = {separacion[c]:.2f}")
ejes[0, 0].legend()
plt.tight_layout()
glue("fig_variables", fig, display=False)
plt.close(fig)
glue("auc_alcohol", f"{separacion['alcohol']:.2f}", display=False)
glue("var2", DICCIONARIO_VINO[top[1]][0].lower(), display=False)
```

```{glue:figure} fig_variables
:name: fig-variables
Distribución por clase de las cuatro variables que mejor separan los vinos buenos (AUC de cada variable usada sola como puntaje).
```

El alcohol es la variable que más separa las clases (AUC {glue:text}`auc_alcohol` usada sola), seguida de {glue:text}`var2`. Las distribuciones se superponen bastante en todas las variables ({numref}`fig-variables`): ninguna basta por sí sola y los modelos deben combinar varias.

```{code-cell} ipython3
:tags: [hide-input]
corr = df[NUMERICAS_VINO + ["quality"]].corr()
nombres = [DICCIONARIO_VINO[c][0] for c in NUMERICAS_VINO] + ["Nota"]
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", [AZUL, "#f0efec", "#e34948"]), vmin=-1, vmax=1)
ax.set_xticks(range(len(nombres)), nombres, rotation=45, ha="right")
ax.set_yticks(range(len(nombres)), nombres)
for i in range(len(nombres)):
    for j in range(len(nombres)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
ax.grid(False)
plt.colorbar(im, ax=ax, fraction=0.04, label="r de Pearson")
ax.set_title("Correlaciones: la densidad depende del alcohol y del azúcar")
plt.tight_layout()
glue("fig_corr", fig, display=False)
plt.close(fig)
c = corr["quality"].drop("quality")
glue("r_alcohol", f"{c['alcohol']:.2f}", display=False)
glue("r_densidad_alcohol", f"{corr.loc['density', 'alcohol']:.2f}", display=False)
```

```{glue:figure} fig_corr
:name: fig-corr
Matriz de correlación de Pearson entre las variables y la nota.
```

La nota se correlaciona sobre todo con el alcohol (r = {glue:text}`r_alcohol`), y la densidad está muy ligada al alcohol (r = {glue:text}`r_densidad_alcohol`), lo que anticipa que ambas se repartirán la importancia en las explicaciones SHAP ({numref}`fig-corr`). Las demás correlaciones con la nota son débiles: la relación es no lineal y con interacciones, lo que favorece a los ensambles de árboles.

## Declaración de los datos

```{include} ../data/DATA_STATEMENT.md
:start-line: 2
```

```{code-cell} ipython3
:tags: [hide-input]
r = resumen_conjuntos()
mostrar(r.rename(columns={"conjunto": "Conjunto", "titulo": "Título", "n": "n", "p_numericas": "Numéricas", "p_categoricas": "Categóricas",
                          "niveles": "Niveles de la respuesta", "umbral": "Umbral del evento", "prevalencia": "Prevalencia", "faltantes": "Faltantes"}),
        {"Prevalencia": "{:.3f}"})
```

(sec-auditoria)=
## Auditoría de fuga de información

Antes de correr los experimentos definitivos se verificó con código cada vía de fuga que enumera el enunciado. La función `src/auditoria.py::tabla_auditoria` reproduce la tabla.

```{code-cell} ipython3
:tags: [hide-input]
from src.auditoria import tabla_auditoria
auditoria, d, vecinos, prep = tabla_auditoria()
mostrar(auditoria.rename(columns={"riesgo": "Riesgo", "verificacion": "Verificación", "resultado": "Resultado", "conclusion": "Conclusión"}))
```

**Gemelos químicos.** Con las particiones de la repetición 1, la distancia mediana de un vino de prueba a su vecino más cercano del entrenamiento (variables estandarizadas) es prácticamente la misma que la de un vino de entrenamiento a su vecino más cercano dentro del entrenamiento, y casi ningún vino de prueba tiene una copia química en el entrenamiento:

```{code-cell} ipython3
:tags: [hide-input]
mostrar(vecinos.rename(columns={"fold": "Pliegue", "prueba_con_gemelo_exacto": "Prueba con gemelo exacto", "prueba_con_vecino_muy_cercano": "Prueba con vecino a < 0.1 DE",
                                "mediana_distancia_prueba": "Distancia mediana prueba → entrenamiento", "mediana_distancia_dentro_entrenamiento": "Distancia mediana dentro del entrenamiento"}),
        {"Prueba con gemelo exacto": "{:.2%}", "Prueba con vecino a < 0.1 DE": "{:.2%}", "Distancia mediana prueba → entrenamiento": "{:.3f}",
         "Distancia mediana dentro del entrenamiento": "{:.3f}"})
```

**Lotes y orden del archivo.** Los residuos del Entregable 2 mostraron autocorrelación en el orden de los archivos originales, que desaparece al desordenar los vinos: vinos consecutivos se parecen más de lo que explica su química, probablemente porque pertenecen al mismo productor o lote. No hay una variable de lote, así que no se puede agrupar de forma exacta. Como verificación, cada modelo se evaluó también con pliegues formados por **bloques de 10 vinos consecutivos** del archivo, que separan entre entrenamiento y prueba a los vinos vecinos:

```{code-cell} ipython3
:tags: [hide-input]
bl = resultado("robustez_bloques.csv").groupby("modelo")["auc"].mean()
ref = resultado("robustez_filas.csv")
ref = ref[ref["estudio"] == "referencia"].groupby("modelo")["auc"].mean()
t = pd.DataFrame({"AUC con pliegues aleatorios": ref, "AUC con bloques del archivo": bl})
t["Diferencia"] = t["AUC con bloques del archivo"] - t["AUC con pliegues aleatorios"]
t = t.sort_values("AUC con pliegues aleatorios", ascending=False)
t.index.name = "Modelo"
caida_media, caida_bof = -t["Diferencia"].mean(), -t.loc[ORIGINAL, "Diferencia"]
if caida_media > 0.005:
    frase = (f"Con bloques del archivo el AUC cae en promedio {caida_media:.3f} (en el Bosque Ordinal Focalizado, {caida_bof:.3f}). "
             "Esa caída estima el optimismo que introducen los vinos «hermanos» cuando quedan a ambos lados de la partición.")
else:
    frase = (f"Con bloques del archivo el AUC cambia en promedio {-caida_media:+.3f} (en el Bosque Ordinal Focalizado, {-caida_bof:+.3f}), "
             "así que no hay indicios de que los vinos vecinos inflen el desempeño.")
glue("frase_bloques", frase, display=False)
mostrar(t.reset_index(), {"AUC con pliegues aleatorios": "{:.4f}", "AUC con bloques del archivo": "{:.4f}", "Diferencia": "{:+.4f}"})
```

{glue:text}`frase_bloques` Es un riesgo residual compartido por todos los modelos (no cambia la comparación entre ellos, que usa siempre las mismas particiones), y se declara como limitación.
