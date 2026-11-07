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

# Evaluación ampliada y robustez

```{code-cell} ipython3
:tags: [remove-cell]
import sys, json, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from src.libro import configurar, mostrar, tabla, resultado, json_tabla, p_texto, AZUL, NARANJA, GRIS, TINTA_2, ORIGINAL, CORTOS
configurar()
v = json_tabla("veredicto.json")
mejor = v["mejor_benchmark"]
NOMBRES_C = {"vino": "Vino", "abalone": "Abalone", "cardiotocografia": "Cardiotocografía", "estudiantes": "Student Performance", "corazon": "Heart Disease"}
```

Los estudios de este capítulo siguen el Cuadro 4 del enunciado. Salvo los conjuntos adicionales, todos usan los hiperparámetros finales de cada modelo y las particiones de la repetición 1; son exploratorios según el protocolo.

## Conjuntos de datos adicionales

```{code-cell} ipython3
:tags: [hide-input]
adic = tabla("adicionales_auc.csv")
ancho = adic.pivot_table(index="modelo", columns="conjunto", values="auc_media")
ancho = ancho[[c for c in NOMBRES_C if c in ancho.columns]].rename(columns=NOMBRES_C)
ancho = ancho.loc[ancho.mean(axis=1).sort_values(ascending=False).index]
ancho.index.name = "Modelo"
mostrar(ancho.reset_index(), {c: "{:.4f}" for c in ancho.columns})
```

AUC medio de los 5 pliegues externos de cada modelo en cada conjunto adicional (validación anidada 5 × 3 con el mismo procedimiento y presupuesto que en vino). Las métricas no se promedian entre conjuntos.

```{code-cell} ipython3
:tags: [hide-input]
ap = tabla("adicionales_pruebas.csv")
mostrar(ap.assign(conjunto=ap["conjunto"].map(NOMBRES_C)).rename(columns={"conjunto": "Conjunto", "mejor_benchmark": "Mejor benchmark", "auc_bof": "AUC BOF",
        "auc_mejor": "AUC mejor benchmark", "bof_mejor_en_media": "¿BOF mejor en media?", "puesto_bof": "Puesto del BOF", "delong_dif": "ΔAUC (DeLong)",
        "delong_p": "DeLong p", "delong_p_holm": "DeLong p (Holm)", "significativamente_peor": "¿Significativamente peor?"}),
        {"AUC BOF": "{:.4f}", "AUC mejor benchmark": "{:.4f}", "ΔAUC (DeLong)": "{:+.4f}", "DeLong p": p_texto, "DeLong p (Holm)": p_texto})
```

```{code-cell} ipython3
:tags: [hide-input]
ec = json_tabla("entre_conjuntos.json")
pares = tabla("entre_conjuntos_pares.csv")
glue("fr_conj_F", f"{ec['friedman']['F_ID']:.2f}", display=False)
glue("fr_conj_p", f"{ec['friedman']['p_ID']:.3f}", display=False)
glue("rango_bof_conj", f"{ec['rangos'][ORIGINAL]:.2f}", display=False)
bm = ec["bayes_vs_mejor_por_conjunto"]
glue("bayes_conj", f"P(mejor) = {bm['p_mejor']:.2f}, P(equivalente) = {bm['p_equivalente']:.2f}, P(peor) = {bm['p_peor']:.2f}", display=False)
mostrar(pares.rename(columns={"benchmark": "Benchmark", "conjuntos_a_favor": "Conjuntos a favor del BOF (de 5)", "diferencia_media": "ΔAUC medio",
                              "wilcoxon_p": "Wilcoxon p", "wilcoxon_p_holm": "Wilcoxon p (Holm)", "bayes_p_peor": "P(peor)", "bayes_p_equivalente": "P(equivalente)",
                              "bayes_p_mejor": "P(mejor)"}),
        {"ΔAUC medio": "{:+.4f}", "Wilcoxon p": p_texto, "Wilcoxon p (Holm)": p_texto, "P(peor)": "{:.2f}", "P(equivalente)": "{:.2f}", "P(mejor)": "{:.2f}"})
```

**Prueba entre conjuntos.** Con los 5 conjuntos como bloques, Friedman con Iman–Davenport da F = {glue:text}`fr_conj_F` (p = {glue:text}`fr_conj_p`) y el rango medio del BOF es {glue:text}`rango_bof_conj`. Con solo 5 conjuntos, Wilcoxon no puede bajar de p = 0.0625 por construcción, así que la evidencia entre conjuntos se lee mejor con la prueba bayesiana de rangos con signo {cite:p}`benavoli2017` (ROPE ±0.01): frente al mejor benchmark de cada conjunto, {glue:text}`bayes_conj`. PENDIENTE: interpretación.

## Sensibilidad a los hiperparámetros propios

```{code-cell} ipython3
:tags: [hide-input]
sens = tabla("robustez_sensibilidad.csv")
ref = tabla("robustez_referencia.csv").groupby("modelo")["auc"].mean()
param = list(dict.fromkeys(sens["parametro"]))
fig, ejes = plt.subplots(2, 3, figsize=(10.5, 5.6))
for ax, p in zip(ejes.ravel(), param):
    d = sens[sens["parametro"] == p]
    x = np.arange(len(d))
    ax.plot(x, d["auc"], marker="o", color=AZUL, lw=2)
    ax.fill_between(x, d["auc"] - d["auc_sd"] / np.sqrt(5), d["auc"] + d["auc_sd"] / np.sqrt(5), color=AZUL, alpha=0.12, lw=0)
    ax.axhline(ref[mejor], color=NARANJA, lw=1.5)
    if mejor != "Random Forest":
        ax.axhline(ref["Random Forest"], color=GRIS, lw=1.2, ls="--")
    ax.set_xticks(x, [{"inf": "∞", "None": "todas"}.get(s, s) for s in d["valor"]], fontsize=8)
    ax.set_title(p, fontsize=10)
ejes[0, 0].set_ylabel("AUC (5 pliegues)")
ejes[1, 0].set_ylabel("AUC (5 pliegues)")
plt.tight_layout()
glue("fig_sens", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_sens
:name: fig-sens
AUC del BOF al mover un hiperparámetro y dejar los demás en sus valores finales (banda: ± 1 error estándar entre pliegues). La línea naranja es el mejor benchmark con sus hiperparámetros finales en los mismos pliegues: por encima de ella el BOF conserva su ventaja. Si el mejor benchmark no es Random Forest, la línea gris discontinua marca a Random Forest, la referencia que fija el protocolo.
```

PENDIENTE: lectura de la sensibilidad.

## Estabilidad por semilla

```{code-cell} ipython3
:tags: [hide-input]
sem = tabla("robustez_semillas.csv")
orden = sem.groupby("modelo")["auc"].mean().sort_values().index
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for i, m in enumerate(orden):
    d = sem[sem["modelo"] == m]["auc"]
    ax.boxplot(d, positions=[i], vert=False, widths=0.5, patch_artist=True, showfliers=False,
               boxprops={"facecolor": AZUL if m == ORIGINAL else "#f0efec", "edgecolor": AZUL if m == ORIGINAL else GRIS},
               medianprops={"color": "black"})
    ax.scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), s=12, color=AZUL if m == ORIGINAL else GRIS, zorder=3)
ax.set_yticks(range(len(orden)), [CORTOS[m] for m in orden])
ax.set_xlabel("AUC medio de 5 pliegues con cada una de 10 semillas")
ax.set_title("Variación por la semilla del modelo")
plt.tight_layout()
glue("fig_semillas", fig, display=False)
plt.close(fig)
de = sem.groupby("modelo")["auc"].std()
glue("de_bof", f"{de[ORIGINAL]:.4f}", display=False)
```

```{glue:figure} fig_semillas
:name: fig-semillas
AUC de los modelos con componentes aleatorios al cambiar su semilla (10 semillas, mismas particiones). La desviación estándar entre semillas del BOF es {glue:text}`de_bof`.
```

PENDIENTE: comparar la variación por semilla con la variación entre particiones ({numref}`fig-cd`).

## Ruido y valores atípicos

```{code-cell} ipython3
:tags: [hide-input]
ruido = tabla("robustez_ruido.csv")
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
titulos = {"variables": "Ruido gaussiano en las variables (σ)", "etiqueta": "Notas movidas ±1 (fracción)", "atipicos": "Atípicos ×10 (fracción)"}
for ax, tipo in zip(ejes, ["variables", "etiqueta", "atipicos"]):
    d = ruido[ruido["parametro"] == tipo]
    for m, g in d.groupby("modelo"):
        g = g.sort_values("valor")
        destacado = m in (ORIGINAL, mejor)
        ax.plot(g["valor"], g["auc"], marker="o", lw=2.2 if destacado else 1, ms=6 if destacado else 3,
                color=AZUL if m == ORIGINAL else (NARANJA if m == mejor else GRIS), alpha=1 if destacado else 0.6,
                label=CORTOS[m] if destacado else None)
    ax.set_title(titulos[tipo], fontsize=10)
    ax.set_xlabel("nivel")
ejes[0].set_ylabel("AUC en prueba")
ejes[0].legend()
plt.tight_layout()
glue("fig_ruido", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_ruido
:name: fig-ruido
AUC al contaminar las variables (entrenamiento y prueba), las notas del entrenamiento o al agregar atípicos al entrenamiento. Azul: BOF; naranja: mejor benchmark; gris: el resto.
```

PENDIENTE: lectura del ruido (en particular, el ruido en la nota afecta directamente la información que usa el BOF).

## Tamaño de muestra y escalabilidad

```{code-cell} ipython3
:tags: [hide-input]
tam = tabla("robustez_tamano.csv")
tam["valor"] = tam["valor"].astype(float)
esc = resultado("robustez_escalabilidad.csv")
fig, ejes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for m, g in tam.groupby("modelo"):
    g = g.sort_values("valor")
    destacado = m in (ORIGINAL, mejor)
    ejes[0].plot(100 * g["valor"], g["auc"], marker="o", lw=2.2 if destacado else 1, color=AZUL if m == ORIGINAL else (NARANJA if m == mejor else GRIS),
                 alpha=1 if destacado else 0.6, label=CORTOS[m] if destacado else None)
ejes[0].set_xlabel("% del entrenamiento usado")
ejes[0].set_ylabel("AUC en prueba")
ejes[0].set_title("Curva de aprendizaje")
ejes[0].legend()
n = esc[esc["eje"] == "n"].groupby(["modelo", "n"])["ajuste_s"].median().reset_index()
for m, g in n.groupby("modelo"):
    destacado = m in (ORIGINAL, mejor)
    pend = np.polyfit(np.log(g["n"]), np.log(g["ajuste_s"]), 1)[0]
    ejes[1].loglog(g["n"], g["ajuste_s"], marker="o", lw=2.2 if destacado else 1, color=AZUL if m == ORIGINAL else (NARANJA if m == mejor else GRIS),
                   alpha=1 if destacado else 0.6, label=f"{CORTOS[m]} (pendiente {pend:.2f})" if destacado else None)
ejes[1].set_xlabel("n (vinos) — escala log")
ejes[1].set_ylabel("Segundos de ajuste — escala log")
ejes[1].set_title("Tiempo de ajuste frente a n")
ejes[1].legend(fontsize=8)
plt.tight_layout()
glue("fig_tam", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_tam
:name: fig-tam
Izquierda: AUC con el 10, 25, 50 y 100 % del entrenamiento (submuestras estratificadas). Derecha: tiempo de ajuste con los hiperparámetros finales al variar n (un solo proceso).
```

PENDIENTE: lectura de la curva de aprendizaje y del costo.

## Cambio de distribución

```{code-cell} ipython3
:tags: [hide-input]
cambio = tabla("robustez_cambio.csv")
t = cambio.pivot_table(index="modelo", columns="entrena", values="auc").rename(columns={"red": "Entrena con tintos → evalúa blancos",
                                                                                       "white": "Entrena con blancos → evalúa tintos"})
t["Referencia (pliegues aleatorios)"] = tabla("robustez_referencia.csv").groupby("modelo")["auc"].mean()
t = t.sort_values("Referencia (pliegues aleatorios)", ascending=False)
t.index.name = "Modelo"
mostrar(t.reset_index(), {c: "{:.4f}" for c in t.columns})
```

PENDIENTE: lectura del cambio de distribución.

(sec-calibracion)=
## Calibración y reglas de puntuación

```{code-cell} ipython3
:tags: [hide-input]
cal = tabla("calibracion.csv")
curvas = tabla("calibracion_curvas.csv")
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1)
for m, g in curvas.groupby("modelo"):
    destacado = m in (ORIGINAL, mejor)
    ax.plot(g["p_media"], g["frecuencia"], marker="o" if destacado else None, lw=2 if destacado else 1,
            color=AZUL if m == ORIGINAL else (NARANJA if m == mejor else GRIS), alpha=1 if destacado else 0.5, label=CORTOS[m] if destacado else None)
ax.set_xlabel("Probabilidad predicha")
ax.set_ylabel("Proporción real de vinos buenos")
ax.set_title("Curvas de confiabilidad (repetición 1)")
ax.legend()
plt.tight_layout()
glue("fig_cal", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_cal
:name: fig-cal
Curvas de confiabilidad con 10 grupos por cuantiles de la probabilidad predicha.
```

```{code-cell} ipython3
:tags: [hide-input]
mostrar(cal.rename(columns={"modelo": "Modelo", "ece": "ECE", "brier": "Brier", "log_loss": "Log-loss", "media_predicha": "Probabilidad media",
                            "prevalencia": "Prevalencia real"}), {"ECE": "{:.4f}", "Brier": "{:.4f}", "Log-loss": "{:.4f}", "Probabilidad media": "{:.3f}",
                                                                  "Prevalencia real": "{:.3f}"})
```

PENDIENTE: lectura de la calibración.

## Cuantificación de la incertidumbre: conjuntos conformes

```{code-cell} ipython3
:tags: [hide-input]
conf = tabla("robustez_conforme.csv")
clase = tabla("robustez_conforme_clase.csv")
t = conf.pivot_table(index="modelo", columns=["alfa", "tipo"], values=["cobertura", "tamano_medio"])
t.columns = [f"{'Cobertura' if a == 'cobertura' else 'Tamaño medio'} {int(100 * (1 - b))} % {c}" for a, b, c in t.columns]
t = t[[c for c in t.columns if "marginal" in c]]
t.index.name = "Modelo"
mostrar(t.reset_index(), {c: "{:.3f}" for c in t.columns})
```

Los conjuntos conformes por partición {cite:p}`angelopoulos2023` usan el 75 % del entrenamiento externo para ajustar y el 25 % para calibrar el umbral de no conformidad; la cobertura se mide en la prueba externa. Con nivel nominal del 90 % y del 95 %, la cobertura empírica debe quedar cerca de esos valores; el tamaño medio dice cuántas veces el conjunto contiene las dos clases (el modelo no decide). PENDIENTE: lectura, incluida la cobertura por clase (Mondrian).

## Análisis de errores

```{code-cell} ipython3
:tags: [hide-input]
err = tabla("errores_subgrupos.csv")
t = err.pivot_table(index=["grupo", "n", "prevalencia"], columns="modelo", values="auc").reset_index()
mostrar(t.rename(columns={"grupo": "Subgrupo", "prevalencia": "Prevalencia"}), {"Prevalencia": "{:.3f}", **{c: "{:.4f}" for c in t.columns if c not in ("grupo", "n", "prevalencia")}})
```

```{code-cell} ipython3
:tags: [hide-input]
from scipy.stats import binomtest
por_nota = pd.read_csv(RAIZ / "experiments" / "resultados" / "tablas" / "errores_por_nota.csv", header=[0, 1], index_col=0)
casos = tabla("errores_casos.csv")
corte = 0.8
casos["acierta_bof"] = np.where(casos["y"] == 1, casos["rango_bof"] > corte, casos["rango_bof"] <= corte)
casos["acierta_rf"] = np.where(casos["y"] == 1, casos["rango_rf"] > corte, casos["rango_rf"] <= corte)
t = pd.DataFrame({"Vinos": por_nota[("BOF", "count")].astype(int), "Probabilidad media BOF": por_nota[("BOF", "mean")],
                  "Probabilidad media RF": por_nota[("RF", "mean")],
                  "Acierta solo el BOF": casos[casos["acierta_bof"] & ~casos["acierta_rf"]].groupby("nota").size(),
                  "Acierta solo RF": casos[~casos["acierta_bof"] & casos["acierta_rf"]].groupby("nota").size()}).fillna(0)
t.index.name = "Nota real"
mostrar(t.reset_index(), {"Probabilidad media BOF": "{:.3f}", "Probabilidad media RF": "{:.3f}", "Acierta solo el BOF": "{:.0f}", "Acierta solo RF": "{:.0f}"})
solo_bof, solo_rf = int(t["Acierta solo el BOF"].sum()), int(t["Acierta solo RF"].sum())
glue("solo_bof", solo_bof, display=False)
glue("solo_rf", solo_rf, display=False)
glue("mcnemar_p", p_texto(binomtest(solo_bof, solo_bof + solo_rf).pvalue if solo_bof + solo_rf else 1.0), display=False)
```

La segunda tabla usa la regla de decisión de una bodega que solo cata el 20 % de vinos con mayor puntaje: un modelo acierta con un vino bueno si lo deja en ese 20 % y con uno no bueno si lo deja fuera. En {glue:text}`solo_bof` vinos acierta solo el BOF y en {glue:text}`solo_rf` solo Random Forest (prueba exacta de McNemar, p = {glue:text}`mcnemar_p`). PENDIENTE: dónde falla el BOF y dónde no supera a los benchmarks.

(sec-interpretabilidad)=
## Interpretabilidad: SHAP del modelo original frente al mejor benchmark

```{code-cell} ipython3
:tags: [hide-input]
shap = pd.read_csv(RAIZ / "data" / "app" / "shap_global.csv")
from src.data import DICCIONARIO_VINO
piv = shap.pivot_table(index="variable", columns="modelo", values="importancia")
cols = [c for c in dict.fromkeys([ORIGINAL, mejor, "Random Forest"]) if c in piv.columns]
piv = piv[cols].div(piv[cols].sum(), axis=1)
piv.index = [DICCIONARIO_VINO[c][0] for c in piv.index]
piv = piv.sort_values(ORIGINAL)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
y = np.arange(len(piv))
alto = 0.8 / len(cols)
for i, (c, color) in enumerate(zip(cols, [AZUL, NARANJA, GRIS])):
    ax.barh(y + (i - (len(cols) - 1) / 2) * alto, piv[c], height=alto, color=color, label=CORTOS[c])
ax.set_yticks(y, piv.index)
ax.set_xlabel("Importancia SHAP relativa (suma = 1)")
ax.set_title("Qué variables usa cada modelo")
ax.legend()
plt.tight_layout()
glue("fig_shap", fig, display=False)
plt.close(fig)
from scipy.stats import spearmanr
glue("mejor_shap", mejor, display=False)
glue("rho_shap", f"{spearmanr(piv[ORIGINAL], piv[mejor]).statistic:.2f}", display=False)
```

```{glue:figure} fig_shap
:name: fig-shap
Importancia SHAP global (media del valor absoluto del aporte, normalizada) del BOF, del mejor benchmark y de Random Forest, calculada sobre los mismos 300 vinos. El BOF se explica con un `TreeExplainer` exacto sobre sus propios árboles (`src/explicaciones.py::arboles_bof`).
```

La correlación de Spearman entre la importancia de las variables del BOF y la del mejor benchmark ({glue:text}`mejor_shap`) es {glue:text}`rho_shap`. PENDIENTE: lectura.
