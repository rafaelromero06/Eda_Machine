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

# Protocolo y validación estadística

```{code-cell} ipython3
:tags: [remove-cell]
import sys, json, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from myst_nb import glue
from src.libro import (configurar, mostrar, tabla, resultado, json_tabla, p_texto, diagrama_cd, cajas, AZUL, NARANJA, GRIS,
                       TINTA_2, ORIGINAL, CORTOS)
configurar()
res = tabla("resumen_modelos.csv")
pr = tabla("pruebas_bof.csv")
fr = json_tabla("friedman.json")
v = json_tabla("veredicto.json")
boot = json_tabla("bootstrap_bca_mejor.json")
mejor = v["mejor_benchmark"]
cv = resultado("cv_filas.csv")
bloques = cv[cv["conjunto"] == "vino"].pivot_table(index=["repeticion", "fold"], columns="modelo", values="auc")
```

## Protocolo pre-registrado

El protocolo (`experiments/protocolo.md`) se congeló el 26 de octubre de 2026 con un único commit (`8bc0c2d`) y el tag anotado `protocolo-e3-v1`, antes de correr los experimentos definitivos. Fija:

- **Hipótesis confirmatorias:** H1 (el BOF supera en AUC medio a la mejor combinación benchmark), H2 (supera a cada uno de los 9 benchmarks), H3 (consistencia en los conjuntos adicionales) y H4 (cada componente aporta, en la ablación). El resto de análisis es exploratorio.
- **Métricas:** AUC-ROC como principal; PR-AUC, Brier, log-loss y F1 como secundarias.
- **Datos:** el conjunto principal y cuatro adicionales elegidos con un criterio fijado antes de evaluar.
- **Particiones y semillas:** validación anidada 5 × 3 repetida con las semillas 42 a 46 (25 bloques pareados); la semilla 42 reproduce las particiones del Entregable 2.
- **Pruebas:** P1 Friedman con Iman–Davenport y Nemenyi; P2 Wilcoxon; P3 *t* corregida de Nadeau–Bengio; P4 DeLong; complementarias: *t* bayesiana correlacionada e intervalo bootstrap BCa; tamaño del efecto con delta de Cliff y *d* pareada; Holm para comparaciones múltiples; α = 0.05.
- **Presupuesto:** 40 evaluaciones de hiperparámetros por pliegue para todos los modelos.
- **Regla de superioridad** (sección 6.4 del enunciado), aplicada literalmente por `src/analisis.py::veredicto`.

El anexo A del protocolo resume las 80 configuraciones de diseño evaluadas antes del congelamiento, todas con validación interna dentro del entrenamiento del primer pliegue externo, y el anexo B guarda la huella SHA-256 de cada archivo de código.

**Desviaciones.** Ninguna afecta a los experimentos confirmatorios:

1. A `src/data.py` se le agregaron al final, después del congelamiento, las funciones de lectura del dashboard (`tabla_app`, `json_app`, `modelo_final` y `metadata_final`). La parte anterior del archivo es idéntica a la congelada: su huella SHA-256 coincide con la del anexo B. Las huellas de los demás archivos del anexo B no cambiaron.
2. Se escribió código nuevo que no interviene en la validación ni en las pruebas: la compactación de los artefactos del BOF para el despliegue (`src/artefactos.py::compactar_bof`), la medición de memoria de la sección 11 del protocolo (`experiments/memoria.py`) y la validación por bloques del archivo, un análisis exploratorio que no estaba previsto y que motivaron los residuos del Entregable 2.
3. Para ahorrar tiempo, las búsquedas finales de hiperparámetros (las que definen los artefactos, sobre los 5320 vinos) corrieron en paralelo con la validación principal: la de CatBoost en un proceso aparte y las de los demás modelos, una tras otra, en otro. Durante ese tramo hubo 12 o 13 procesos en lugar de 11 (y por momentos hasta 17, mientras se ensayaban los scripts de análisis con resultados simulados), así que los tiempos medidos en ese periodo están algo inflados. Esto afecta al reporte de cómputo, no a las métricas: las búsquedas finales no usan la validación ni sus resultados.

## Supuestos de las pruebas

```{code-cell} ipython3
:tags: [hide-input]
dif = (bloques[ORIGINAL] - bloques[mejor]).to_numpy()
sw = stats.shapiro(dif)
asim = stats.skew(dif)
glue("shapiro_p", f"{sw.pvalue:.3f}", display=False)
glue("asimetria", f"{asim:.2f}", display=False)
glue("mejor", mejor, display=False)
```

- **Wilcoxon (P2)** supone diferencias pareadas independientes y distribuidas simétricamente alrededor de la mediana. La asimetría de las 25 diferencias de AUC frente a {glue:text}`mejor` es {glue:text}`asimetria`.
- ***t* corregida (P3)** supone normalidad aproximada de las diferencias; la prueba de Shapiro–Wilk da p = {glue:text}`shapiro_p`. La independencia **no** se cumple, porque los 25 bloques comparten vinos (cada vino está en 4 de los 5 entrenamientos de cada repetición). Por eso se usa la corrección de {cite:t}`nadeau2003`, que infla la varianza de la media por (1/J + n<sub>prueba</sub>/n<sub>entrenamiento</sub>) = 1/25 + 0.25 en lugar de 1/25; la *t* sin corregir sería anticonservadora {cite:p}`dietterich1998,bouckaert2004`.
- **DeLong (P4)** supone casos independientes y compara dos AUC calculados sobre los mismos vinos; se aplica a las predicciones fuera de muestra de la repetición 1 (cada vino predicho una vez, por un modelo que no lo vio). Los vinos son intercambiables (no hay estructura temporal), con la salvedad de los posibles lotes del {ref}`sec-auditoria`.
- **Friedman y Nemenyi (P1)** suponen bloques independientes; los 25 bloques comparten vinos, así que P1 se interpreta como un procedimiento global aproximado {cite:p}`demsar2006`.
- Las pruebas de Diebold–Mariano, Giacomini–White, MCS y SPA suponen orden temporal; como el problema es de corte transversal, se usan las de la columna correspondiente del enunciado.

## Prueba global: Friedman y diferencia crítica

```{code-cell} ipython3
:tags: [hide-input]
rangos = tabla("rangos_friedman.csv").set_index(tabla("rangos_friedman.csv").columns[0])["rango_medio"]
fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
diagrama_cd(ejes[0], rangos, fr["cd"])
ejes[0].set_title(f"Nemenyi: CD = {fr['cd']:.2f}")
cajas(ejes[1], bloques)
ejes[1].set_xlabel("AUC en cada bloque")
ejes[1].set_title("AUC de los 25 bloques")
plt.tight_layout()
glue("fig_cd", fig, display=False)
plt.close(fig)
glue("F_ID", f"{fr['F_ID']:.1f}", display=False)
glue("p_ID", f"{fr['p_ID']:.1e}", display=False)
glue("W", f"{fr['W']:.2f}", display=False)
glue("rango_bof", f"{rangos[ORIGINAL]:.2f}", display=False)
glue("rango_mejor", f"{rangos[mejor]:.2f}", display=False)
```

```{glue:figure} fig_cd
:name: fig-cd
Izquierda: rangos medios de los 10 modelos en los 25 bloques y diferencia crítica de Nemenyi (α = 0.05); los modelos dentro de la franja azul no se distinguen del mejor. Derecha: AUC de cada bloque.
```

La prueba de Friedman con la corrección de Iman–Davenport rechaza que los 10 modelos sean iguales (F = {glue:text}`F_ID`, p = {glue:text}`p_ID`), con un tamaño de efecto W de Kendall de {glue:text}`W`. El rango medio del BOF es {glue:text}`rango_bof` y el del mejor benchmark ({glue:text}`mejor`), {glue:text}`rango_mejor`. PENDIENTE: lectura del diagrama.

## Pruebas pareadas frente a cada benchmark

```{code-cell} ipython3
:tags: [hide-input]
nombres = {"benchmark": "Benchmark", "auc_benchmark": "AUC benchmark", "diferencia_media": "ΔAUC medio", "bloques_a_favor": "Bloques a favor (de 25)",
           "nemenyi_p": "Nemenyi p", "wilcoxon_p": "Wilcoxon p", "wilcoxon_p_holm": "Wilcoxon p (Holm)", "t_p": "t corregida p",
           "t_p_holm": "t corregida p (Holm)", "delong_dif": "ΔAUC DeLong (rep. 1)", "delong_p": "DeLong p", "delong_p_holm": "DeLong p (Holm)",
           "delta_cliff": "δ de Cliff", "magnitud_cliff": "Magnitud", "d_cohen_pareada": "d pareada"}
t = pr[list(nombres)].rename(columns=nombres)
formato = {n: p_texto for n in ["Nemenyi p", "Wilcoxon p", "Wilcoxon p (Holm)", "t corregida p", "t corregida p (Holm)", "DeLong p", "DeLong p (Holm)"]}
formato.update({"AUC benchmark": "{:.4f}", "ΔAUC medio": "{:+.4f}", "ΔAUC DeLong (rep. 1)": "{:+.4f}", "δ de Cliff": "{:+.2f}", "d pareada": "{:+.2f}"})
mostrar(t, formato)
```

Se hicieron 27 comparaciones confirmatorias (3 pruebas pareadas × 9 benchmarks) más la comparación global de Friedman. Los p-valores ajustados por Holm se calculan dentro de cada prueba sobre las 9 comparaciones; como sensibilidad, la tabla completa en `experiments/resultados/tablas/pruebas_bof.csv` también trae Holm sobre las 27.

## Tabla de pruebas frente al mejor benchmark

```{code-cell} ipython3
:tags: [hide-input]
formal = tabla("tabla_pruebas.csv")
mostrar(formal.rename(columns={"prueba": "Prueba", "hipotesis_nula": "Hipótesis nula", "supuestos": "Supuestos verificados", "estadistico": "Estadístico",
                               "p_crudo": "p crudo", "p_ajustado": "p ajustado", "efecto": "Tamaño del efecto", "conclusion": "Conclusión"}),
        {"p crudo": p_texto, "p ajustado": p_texto})
```

```{code-cell} ipython3
:tags: [hide-input]
fila = pr.set_index("benchmark").loc[mejor]
glue("bayes_mejor", f"{fila['bayes_p_mejor']:.2f}", display=False)
glue("bayes_equiv", f"{fila['bayes_p_equivalente']:.2f}", display=False)
glue("bayes_peor", f"{fila['bayes_p_peor']:.2f}", display=False)
glue("bca", f"{boot['diferencia']:+.4f} [{boot['ic_inf']:+.4f}, {boot['ic_sup']:+.4f}]", display=False)
rep = tabla("delong_repeticiones.csv")
r_m = rep[rep["benchmark"] == mejor]
glue("delong_reps", f"{int((r_m['p_holm'] < 0.05).sum())} de {len(r_m)}", display=False)
```

**Pruebas complementarias.** La prueba *t* bayesiana correlacionada {cite:p}`corani2015` con una región de equivalencia práctica de ±0.005 de AUC da P(BOF mejor) = {glue:text}`bayes_mejor`, P(equivalentes) = {glue:text}`bayes_equiv` y P(BOF peor) = {glue:text}`bayes_peor`. El intervalo bootstrap BCa {cite:p}`efron1987` de la diferencia de AUC en la repetición 1 es {glue:text}`bca`. Repitiendo DeLong en cada una de las 5 repeticiones, la diferencia frente a {glue:text}`mejor` es significativa (Holm) en {glue:text}`delong_reps`.

## Significancia práctica

```{code-cell} ipython3
:tags: [hide-input]
from sklearn.metrics import roc_curve
pred = resultado("cv_predicciones.csv")
p0 = pred[(pred["conjunto"] == "vino") & (pred["repeticion"] == 0)]
ancho = p0.pivot_table(index="idx", columns="modelo", values="puntaje")
y0 = p0.drop_duplicates("idx").set_index("idx").loc[ancho.index, "y_evento"].to_numpy()
n_buenos, n_vinos = int(y0.sum()), len(y0)
k = int(round(0.2 * n_vinos))
filas = []
for m in [ORIGINAL, mejor]:
    s = ancho[m].to_numpy()
    fpr, tpr, _ = roc_curve(y0, s)
    filas.append({"Modelo": m, "Sensibilidad con 10 % de falsos positivos": np.interp(0.10, fpr, tpr),
                  "Vinos buenos detectados (de " + str(n_buenos) + ")": np.interp(0.10, fpr, tpr) * n_buenos,
                  f"Vinos buenos entre los {k} mejor puntuados": int(y0[np.argsort(-s, kind="stable")[:k]].sum())})
practica = pd.DataFrame(filas)
mostrar(practica, {"Sensibilidad con 10 % de falsos positivos": "{:.3f}", "Vinos buenos detectados (de " + str(n_buenos) + ")": "{:.0f}"})
dif_vinos = practica.iloc[0, 2] - practica.iloc[1, 2]
dif_top = practica.iloc[0, 3] - practica.iloc[1, 3]
glue("n_buenos", n_buenos, display=False)
glue("k_top", k, display=False)
glue("dif_vinos", f"{dif_vinos:+.0f}", display=False)
glue("dif_top", f"{dif_top:+d}", display=False)
```

La tabla traduce la diferencia a vinos con las predicciones fuera de muestra de la repetición 1 (los 5320 vinos, {glue:text}`n_buenos` buenos). Con la misma tasa de falsos positivos (10 % de los vinos no buenos marcados como buenos), la diferencia entre el BOF y {glue:text}`mejor` es de {glue:text}`dif_vinos` vinos buenos detectados; si una bodega solo pudiera catar los {glue:text}`k_top` vinos con mayor puntaje (el 20 %), la diferencia sería de {glue:text}`dif_top` vinos buenos. PENDIENTE: juicio sobre la relevancia.

(sec-veredicto)=
## Veredicto sobre la superioridad (R3)

```{code-cell} ipython3
:tags: [hide-input]
nombres = {"P1_nemenyi": "P1 Nemenyi", "P2_wilcoxon": "P2 Wilcoxon", "P3_t_corregida": "P3 t corregida", "P4_delong": "P4 DeLong"}
mostrar(pd.DataFrame([
    {"Condición": "1. AUC medio mayor que los 9 benchmarks", "Cumple": "sí" if v["condicion_1_media_mayor_que_todos"] else "no",
     "Detalle": f"BOF {v['auc_bof']:.4f} frente a {v['auc_mejor_benchmark']:.4f} ({mejor})"},
    {"Condición": "2. Al menos 3 de 4 pruebas rechazan a favor del BOF (p ajustado < 0.05)", "Cumple": "sí" if v["condicion_2_tres_pruebas"] else "no",
     "Detalle": ", ".join(f"{nombres[k]}: {'rechaza' if ok else 'no rechaza'}" for k, ok in v["pruebas_que_rechazan"].items())},
    {"Condición": "3. Efecto no despreciable (|δ| ≥ 0.147 o |d| ≥ 0.2)", "Cumple": "sí" if v["condicion_3_efecto_no_despreciable"] else "no",
     "Detalle": f"δ de Cliff = {v['delta_cliff']:.2f}; d pareada = {v['d_cohen_pareada']:.2f}"},
    {"Condición": "4. Consistencia en los conjuntos adicionales", "Cumple": "sí" if v["condicion_4_consistencia"] else "no",
     "Detalle": f"mejor en {v['conjuntos_donde_bof_gana']} de 4; significativamente peor en {v['conjuntos_significativamente_peor']}"},
    {"Condición": "Superior (R3)", "Cumple": "sí" if v["superior"] else "no", "Detalle": "todas las condiciones a la vez"},
]))
```

PENDIENTE: redacción del veredicto.
