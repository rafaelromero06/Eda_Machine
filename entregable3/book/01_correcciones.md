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

# Trazabilidad de las correcciones del Entregable 2

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import pandas as pd
from src.libro import configurar, mostrar
configurar()
E2 = RAIZ.parent / "modelado_entregable2" / "modelado" / "experiments" / "tablas"
```

La retroalimentación del Entregable 2 reconoció la base del trabajo (136 combinaciones evaluadas con validación anidada 5 × 3) y señaló lo que faltaba: interpretabilidad con SHAP y LIME, optimización computacional con análisis de complejidad y perfilamiento, calibración, análisis de residuos, pruebas de DeLong y MCS, tamaños de efecto, versiones fijadas del entorno y una tabla maestra consolidada. Todas esas observaciones se atendieron con código y resultados nuevos en el repositorio del Entregable 2 (`modelado_entregable2/modelado/notebooks/02` a `06`), y el Entregable 3 las extiende al modelo original.

## Tabla de trazabilidad

| Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|
| No hay interpretabilidad con SHAP | SHAP exacto (TreeExplainer) sobre las 5320 predicciones fuera de muestra de Random Forest y XGBoost; en el Entregable 3, SHAP del modelo original con un explicador propio de sus árboles | {ref}`sec-corr-interpretabilidad`; {ref}`sec-interpretabilidad` | {numref}`fig-e2-shap`; tabla `tabla21_importancia_shap.csv`; figura SHAP del BOF |
| No hay LIME | LIME en tres casos (verdadero positivo, falso negativo, falso positivo), estabilidad con 30 semillas y acuerdo global con SHAP en 200 vinos | {ref}`sec-corr-interpretabilidad` | {numref}`fig-e2-lime`; `tabla22_lime_vs_shap_global.csv` |
| No hay análisis de complejidad | Complejidad teórica de los 7 modelos y del remuestreo, contrastada con el exponente empírico del tiempo frente a n; complejidad demostrada del BOF | {ref}`sec-corr-costo`; capítulo 5 (complejidad) | {numref}`fig-e2-escala`; `tabla23_complejidad_teorica.csv`, `tabla24_escalamiento_empirico.csv` |
| No hay perfilamiento | `cProfile` de una evaluación de la búsqueda, reparto del tiempo por etapa, paralelismo (ley de Amdahl) y orden de la cola | {ref}`sec-corr-costo` | {numref}`fig-e2-perfil`; `tabla25_perfil_por_etapa.csv`, `tabla26_planificacion_tareas.csv` |
| No hay calibración | Curvas de confiabilidad, ECE, Brier, pendiente de calibración y recalibración (Platt e isotónica); en el Entregable 3, calibración de los 10 modelos comparados y calibración fuera de bolsa propia del BOF | {ref}`sec-corr-calibracion`; {ref}`sec-calibracion` | {numref}`fig-e2-calibracion`; `tabla10_clf_calibracion.csv`, `tabla11_clf_recalibracion.csv` |
| No hay análisis de residuos | Normalidad (Jarque–Bera), varianza (Breusch–Pagan, White), autocorrelación (Ljung–Box, Durbin–Watson) e independencia (BDS) de los residuos de la regresión, en el orden del archivo y en orden aleatorio | {ref}`sec-corr-residuos` | {numref}`fig-e2-residuos`; `tabla12_reg_pruebas_residuos.csv` |
| La estadística se limita a Friedman y Nemenyi: falta DeLong | DeLong por pares (algoritmo de Sun y Xu) entre los 7 modelos y para balanceo y optimizador; en el Entregable 3 es la prueba P4 del protocolo | {ref}`sec-corr-estadistica`; capítulo 6 | {numref}`fig-e2-delong`; `tabla15_delong_pares_de_modelos.csv`; tabla de pruebas del capítulo 6 |
| Falta MCS | Model Confidence Set (Hansen, Lunde y Nason) con el Brier por vino en clasificación y el error cuadrático en regresión | {ref}`sec-corr-estadistica` | {numref}`fig-e2-mcs`; `tabla18_conjunto_confianza_modelos.csv` |
| Faltan tamaños de efecto | W de Kendall de cada Friedman, rango-biserial y d de Cohen pareados con intervalos bootstrap; en el Entregable 3, delta de Cliff | {ref}`sec-corr-estadistica`; capítulo 6 | {numref}`fig-e2-efectos`; `tabla14_friedman_w_kendall.csv`, `tabla19_tamanos_de_efecto.csv` |
| Las versiones del entorno no están fijadas | `requirements.txt` y `environment.yml` con versiones exactas (`==`) y `entorno.txt` con los 80 paquetes y el equipo; en el Entregable 3, `requirements.txt` (dashboard), `requirements-dev.txt`, `.python-version` y `Dockerfile` | capítulo 10 | archivos del repositorio |
| Falta una tabla maestra consolidada | `tabla_maestra.csv`: 136 combinaciones × 36 columnas con métricas, calibración, pruebas, MCS, costo e hiperparámetros; en el Entregable 3 se amplía con el modelo original, los métodos recientes y los conjuntos adicionales | capítulo 10 | `tabla_maestra.csv`, `experiments/resultados/tabla_maestra_e3.csv` |
| (Entregable 3) Auditoría de fuga | Verificación programática de cada riesgo de fuga y validación por bloques del archivo | {ref}`sec-auditoria` | tabla de auditoría del capítulo 2 |

(sec-corr-interpretabilidad)=
## Interpretabilidad: SHAP y LIME

```{figure} figuras_e2/fig22_shap_importancia_global.png
:name: fig-e2-shap
:width: 90%
Importancia SHAP global de Random Forest en clasificación y regresión (Entregable 2). El alcohol domina; le siguen la densidad, la acidez volátil y los cloruros.
```

```{figure} figuras_e2/fig26_lime_estabilidad.png
:name: fig-e2-lime
:width: 90%
Estabilidad de LIME: con 1000 muestras, las tres variables principales cambian entre semillas; con 5000 se estabilizan. SHAP, en cambio, es exacto y determinista para árboles.
```

(sec-corr-costo)=
## Complejidad y perfilamiento

```{figure} figuras_e2/fig29_escalamiento_empirico.png
:name: fig-e2-escala
:width: 90%
Tiempo de entrenamiento y de predicción frente al tamaño de la muestra (escala log-log). La pendiente es el exponente empírico de la complejidad; SVM es el único que crece casi como $n^2$.
```

```{figure} figuras_e2/fig30_tiempo_por_etapa.png
:name: fig-e2-perfil
:width: 90%
Reparto del tiempo de una evaluación de la búsqueda según `cProfile`: más del 70 % se va en el código compilado de cada librería; el preprocesamiento y el remuestreo pesan poco.
```

(sec-corr-calibracion)=
## Calibración

```{figure} figuras_e2/fig14_clf_curvas_confiabilidad.png
:name: fig-e2-calibracion
:width: 90%
Curvas de confiabilidad de la mejor combinación de cada modelo antes y después de recalibrar (Entregable 2).
```

(sec-corr-residuos)=
## Residuos de la regresión

```{figure} figuras_e2/fig17_reg_diagnostico_residuos.png
:name: fig-e2-residuos
:width: 90%
Residuos de la regresión recomendada: varianza no constante, colas pesadas y autocorrelación en el orden del archivo que desaparece al desordenar los vinos (posibles lotes).
```

La autocorrelación en el orden del archivo motivó una verificación nueva en el Entregable 3: la validación por bloques consecutivos del archivo ({ref}`sec-auditoria`).

(sec-corr-estadistica)=
## DeLong, MCS y tamaños de efecto

```{figure} figuras_e2/fig19_diferencias_frente_a_random_forest.png
:name: fig-e2-delong
:width: 90%
Diferencia de AUC (DeLong) y de error cuadrático (Diebold–Mariano con corrección de Harvey, Leybourne y Newbold) de cada modelo frente a Random Forest, con intervalos del 95 % y p-valores ajustados por Holm.
```

```{figure} figuras_e2/fig20_conjunto_confianza_modelos.png
:name: fig-e2-mcs
:width: 90%
Model Confidence Set con α = 0.10: en clasificación quedan configuraciones de Random Forest con `class_weight` o sin balanceo y una de XGBoost; ninguna con SMOTE o ADASYN.
```

```{figure} figuras_e2/fig21_tamanos_de_efecto.png
:name: fig-e2-efectos
:width: 90%
Tamaños de efecto (rango-biserial pareado e intervalos bootstrap) de las comparaciones principales del Entregable 2.
```

## ¿Cambiaron los resultados?

Las correcciones agregaron análisis; no cambiaron ningún resultado del Entregable 2. Se verificó que el código del Entregable 3 reproduce exactamente el Entregable 2: con las mismas particiones (semilla 42), los 7 modelos benchmark dan el mismo AUC en el pliegue externo 1 (diferencia 0) y los mismos hiperparámetros.

Sí cambió un criterio, como pide el enunciado del Entregable 3: la mejor combinación de cada modelo se elige ahora por el **AUC medio del bucle externo**, mientras que el Entregable 2 la elegía por el puntaje de la validación interna. El cambio altera la combinación elegida en {glue:text}`n_cambios` de los 7 modelos y no altera el orden de los modelos. En Naive Bayes hay un empate exacto de AUC entre no balancear y `class_weight`, porque reponderar las clases solo cambia la probabilidad a priori y no el orden de los vinos; se conservó `class_weight`.

```{code-cell} ipython3
:tags: [hide-input]
from myst_nb import glue
from src.benchmarks import BENCHMARKS_CLASE, MEJORES
log = pd.read_csv(RAIZ.parent / "modelado_entregable2" / "modelado" / "experiments" / "logs" / "resultados_clasificacion_p40.csv")
g = log.groupby(["modelo", "balanceo", "optimizador"]).agg(auc=("auc", "mean"), interno=("score_interno", "mean")).reset_index()
por_interno = g.sort_values("interno").groupby("modelo").tail(1).set_index("modelo")
filas = []
for m in BENCHMARKS_CLASE:
    b, o = MEJORES[m]
    auc_e3 = g[(g["modelo"] == m) & (g["balanceo"] == b) & (g["optimizador"] == o)]["auc"].iloc[0]
    filas.append({"Modelo": m, "Elegida por AUC externo (E3)": f"{b} · {o}", "AUC externo": auc_e3,
                  "Elegida por score interno (E2)": f"{por_interno.loc[m, 'balanceo']} · {por_interno.loc[m, 'optimizador']}",
                  "AUC externo de la elegida en E2": por_interno.loc[m, "auc"],
                  "cambia": (b, o) != (por_interno.loc[m, "balanceo"], por_interno.loc[m, "optimizador"])})
comp = pd.DataFrame(filas).sort_values("AUC externo", ascending=False)
glue("n_cambios", int(comp["cambia"].sum()), display=False)
mostrar(comp.drop(columns="cambia"), {"AUC externo": "{:.4f}", "AUC externo de la elegida en E2": "{:.4f}"})
```

Elegir por el AUC externo favorece un poco a los benchmarks (la misma partición sirve para elegir y para medir), así que es una elección conservadora para la comparación con el modelo original: el modelo original no se elige así, sino con su propia búsqueda interna.
