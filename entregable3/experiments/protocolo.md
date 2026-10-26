# Protocolo experimental pre-registrado — Entregable 3

**Proyecto:** Wine Quality — Machine Learning, Universidad del Norte
**Autores:** Roger Daza, Matteo Del Sole, Rafael Romero
**Versión:** 1.0 · **Fecha de congelamiento:** 2026-10-26 · **Etiqueta de Git:** `protocolo-e3-v1`

Este documento fija, antes de correr los experimentos definitivos, las hipótesis, métricas, datos, particiones, presupuesto, pruebas y reglas de decisión. Cualquier cambio posterior se declara como desviación en el informe; los análisis que no estén aquí son exploratorios y no pueden sustentar por sí solos la condición R3.

## 1. Tarea y modelo original

- **Tarea:** clasificación binaria del evento *vino bueno*, `quality >= 7`, a partir de las 11 variables fisicoquímicas y el tipo de vino (tinto o blanco). Conjunto: 5320 vinos tras quitar 1177 duplicados exactos; prevalencia del evento 18.97 %.
- **Modelo original:** Bosque Ordinal Focalizado (BOF), implementado desde cero en `src/original_model.py` con numpy y scipy como únicas herramientas numéricas. Aprende todos los indicadores acumulados `1{quality >= t}` con árboles extremadamente aleatorizados de salida múltiple, elige cada partición con una impureza acumulada ponderada por un núcleo centrado en el umbral del evento, puntúa con una combinación focalizada de las probabilidades acumuladas de umbrales vecinos y calibra con una regresión logística sobre los puntajes fuera de bolsa.
- **Historia del diseño:** antes de este protocolo se evaluaron 5 rondas de diseño (80 configuraciones en total), siempre con validación interna 3 × 3 repetida (semilla 7) dentro del entrenamiento del pliegue externo 1 de la repetición 1 en vino. Ningún pliegue externo de prueba ni ningún conjunto adicional se usó en el diseño. El resumen está en el anexo A.

## 2. Hipótesis

**Confirmatorias (sustentan R3):**

- **H1.** En Wine Quality, el BOF tiene un AUC-ROC medio en el bucle externo mayor que el de la mejor combinación benchmark.
- **H2.** El BOF tiene un AUC medio mayor que la mejor combinación de cada uno de los 7 modelos base y que los 2 métodos recientes (LightGBM y CatBoost).
- **H3.** En los conjuntos adicionales, el BOF tiene un AUC medio mayor que el mejor benchmark de cada conjunto en al menos la mitad de ellos (2 de 4) y no es significativamente peor que ese benchmark en ninguno.

**Confirmatoria de componentes (ablación):**

- **H4.** Cada componente del BOF aporta AUC: quitar la agregación focalizada, usar agregación uniforme, usar partición binaria o quitar la información ordinal reduce el AUC respecto del modelo completo.

**Exploratorias:** calidad probabilística, sensibilidad a hiperparámetros, estabilidad por semilla, ruido y atípicos, tamaño de muestra y escalabilidad, cambio de distribución, conjuntos de predicción conformes, análisis de errores, interpretabilidad y costo computacional.

## 3. Métricas

- **Principal:** AUC-ROC del evento en cada pliegue externo.
- **Secundarias:** PR-AUC; puntaje de Brier y log-loss (solo modelos que entregan probabilidades, SVM queda excluido); exactitud, precisión, sensibilidad y F1 con umbral 0.5; tiempo de búsqueda, de ajuste y de predicción por caso.

## 4. Datos

| Conjunto | Fuente | n | Variables | Respuesta ordinal | Evento | Prevalencia |
|---|---|---|---|---|---|---|
| Vino (principal) | UCI 186, Cortez et al. (2009) | 5320 | 11 num. + tipo | quality (3–9) | quality ≥ 7 | 0.190 |
| Abalone | UCI 1, doi 10.24432/C55C7W | 4177 | 7 num. + sexo | anillos (1–29) | anillos ≥ 11 | 0.346 |
| Cardiotocografía | UCI 193, doi 10.24432/C51S4N | 2115 | 21 num. | NSP (1–3) | NSP ≥ 2 | 0.221 |
| Student Performance (portugués) | UCI 320, doi 10.24432/C5TG7T | 649 | 13 num. + 17 cat. | G3 (0–19) | G3 ≥ 10 | 0.846 |
| Heart Disease (Cleveland) | UCI 45, doi 10.24432/C52P4X | 303 | 9 num. + 4 cat. | num (0–4) | num ≥ 1 | 0.459 |

**Criterio de selección de los conjuntos adicionales** (fijado sin conocer el desempeño de ningún modelo en ellos): repositorio público (UCI), tabulares, clasificación binaria definida al cortar una respuesta ordinal que viene en los propios datos (el escenario para el que se diseñó el método), sin estructura temporal ni de grupos que exija otra validación, n entre 300 y 6000, y naturaleza distinta entre sí: dominio (biología, medicina, educación), tamaño (303 a 4177), dimensión (8 a 30 variables), número de niveles (3 a 28) y prevalencia (0.22 a 0.85). El umbral de cada evento es el habitual en la literatura del conjunto (aprobación con 10 en Student Performance, presencia de enfermedad en Heart Disease, registro no normal en Cardiotocografía y la clase superior de la partición usual 1–8 / 9–10 / 11+ en Abalone).

**Preprocesamiento:** se eliminan filas duplicadas exactas; se excluyen variables que son otra respuesta (G1 y G2 en Student Performance, CLASS en Cardiotocografía). Dentro de cada pliegue: imputación por mediana (numéricas) o moda (categóricas) cuando hay faltantes, escalado estándar y one-hot. Todo ajuste con parámetros aprendidos ocurre solo con el entrenamiento del pliegue.

## 5. Particiones, repeticiones y semillas

- **Vino:** validación anidada 5 × 3 repetida 5 veces con semillas 42, 43, 44, 45 y 46. Pliegues externos `StratifiedKFold(5, shuffle=True, random_state=semilla)` sobre el evento; internos `StratifiedKFold(3, shuffle=True, random_state=semilla)` sobre el entrenamiento externo. La repetición con semilla 42 reproduce exactamente las particiones del Entregable 2. Son 25 bloques pareados.
- **Conjuntos adicionales:** la misma validación anidada 5 × 3 con una repetición (semilla 42).
- Todos los modelos usan las mismas particiones externas e internas. Semilla del optimizador: `semilla + pliegue`. Semilla de los estimadores: 42.

## 6. Modelos comparados y presupuesto

- **Benchmarks vistos en clase** (mejor combinación de cada modelo por AUC medio externo del Entregable 2): Random Forest + class_weight + Grid Search; XGBoost + SMOTE + Bayesiano; SVM + class_weight + Random Search; KNN sin balanceo + Bayesiano; Regresión Logística sin balanceo + Bayesiano; Árbol de decisión sin balanceo + Random Search; Naive Bayes + class_weight + Bayesiano. Espacios de búsqueda idénticos al Entregable 2.
- **Métodos recientes:** LightGBM (Ke et al., 2017) y CatBoost (Prokhorenkova et al., 2018), con optimización bayesiana (TPE). La ponderación de clases es un hiperparámetro más y la primera evaluación usa sus hiperparámetros por defecto.
- **Modelo original:** BOF con optimización bayesiana (TPE). Espacio: `n_estimators` 100–500, `max_features` ∈ {sqrt, log2, None}, `min_samples_leaf` 1–20 (log), `ancho_agregacion` 0.1–5 (log), `ancho_particion` 0.1–10 (log), `desplazamiento` −1 a 1. No se encola ninguna configuración inicial.
- **Presupuesto:** 40 evaluaciones de hiperparámetros por pliegue externo para todos los modelos (la grilla adaptativa de Grid Search queda en ≤ 40 puntos), cada una con validación interna de 3 pliegues; objetivo interno AUC-ROC. Se reporta el número real de evaluaciones y el tiempo consumido por modelo.

## 7. Pruebas de hipótesis (H1 y H2)

Sobre los 25 bloques pareados de vino:

- **P1. Friedman con corrección de Iman–Davenport** sobre los 10 modelos y post-hoc de **Nemenyi** con diagrama de diferencia crítica (procedimiento global).
- **P2. Wilcoxon de rangos con signo**, bilateral, BOF frente a cada uno de los 9 benchmarks.
- **P3. t corregida de Nadeau–Bengio** para validación repetida (J = 25, n_prueba/n_entrenamiento = 0.25), bilateral, BOF frente a cada benchmark.
- **P4. DeLong** sobre las 5320 predicciones fuera de muestra de la repetición 1 (las particiones del Entregable 2), bilateral, BOF frente a cada benchmark. Como sensibilidad se repite en las repeticiones 2 a 5.
- **Complementarias:** prueba t bayesiana correlacionada (ρ = 0.2, ROPE = ±0.005 de AUC) e intervalo bootstrap BCa (B = 2000, remuestreo pareado y estratificado) de la diferencia de AUC en la repetición 1.
- **Tamaño del efecto:** delta de Cliff entre los 25 AUC de cada modelo (umbrales de Romano et al.: < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano, en adelante grande) y d de Cohen pareada.
- **Comparaciones múltiples:** α = 0.05. En P2, P3 y P4 se ajusta por Holm dentro de cada prueba sobre las 9 comparaciones. Se reportan p crudos y ajustados y el total de comparaciones; como sensibilidad, Holm sobre las 27 comparaciones juntas.

**Regla de superioridad (R3).** El BOF se declara superior si se cumplen a la vez:

1. Su AUC medio en los 25 bloques es mayor que el de cada uno de los 9 benchmarks (en particular, que el del mejor).
2. Frente al mejor benchmark (el de mayor AUC medio en los 25 bloques), al menos tres de las cuatro pruebas rechazan la igualdad con p ajustado < 0.05 y a favor del BOF: P1 (comparación de Nemenyi del par), P2, P3 y P4.
3. El tamaño del efecto frente al mejor benchmark no es despreciable: |δ de Cliff| ≥ 0.147 o |d| ≥ 0.2.
4. H3 se cumple.

## 8. Ablación (H4)

Con los hiperparámetros que eligió la búsqueda del BOF en cada pliegue externo (sin volver a optimizar), se reentrena en el mismo entrenamiento externo y se evalúa en la misma prueba: completo; sin agregación (`ancho_agregacion = 0`); agregación uniforme (`ancho_agregacion = inf`); partición binaria (`ancho_particion = 0`); sin información ordinal (se entrena con el evento binario); sin calibración (solo afecta las métricas de probabilidad). Pruebas: P2, P3 y P4 del modelo completo frente a cada variante, Holm sobre las variantes dentro de cada prueba. También se reporta en los conjuntos adicionales como evidencia exploratoria.

## 9. Conjuntos adicionales (H3)

- En cada conjunto, AUC medio de los 5 pliegues de los 10 modelos con el mismo procedimiento y presupuesto.
- **Mejor benchmark del conjunto:** el de mayor AUC medio. Prueba de DeLong sobre las predicciones fuera de muestra (BOF frente a ese benchmark), Holm sobre los 4 conjuntos. "Significativamente peor" = p ajustado < 0.05 con diferencia negativa.
- **Entre conjuntos (5 incluido vino):** Friedman con Iman–Davenport sobre los 10 modelos, Wilcoxon–Holm por pares del BOF frente a cada benchmark y prueba bayesiana de rangos con signo (ROPE = ±0.01) frente a Random Forest y frente al mejor benchmark de cada conjunto. Las métricas no se promedian entre conjuntos.

## 10. Estudios de robustez y calidad probabilística (exploratorios)

Con los hiperparámetros finales de cada modelo (búsqueda con el mismo presupuesto sobre los 5320 vinos) y las particiones de la repetición 1:

- **Sensibilidad:** AUC del BOF al variar uno a uno `ancho_agregacion`, `ancho_particion`, `desplazamiento`, `max_features`, `min_samples_leaf` y `n_estimators`, con la región donde supera al Random Forest.
- **Estabilidad:** 10 semillas de los estimadores (0 a 9) en los modelos estocásticos; dispersión del AUC y diagrama de cajas, junto con la dispersión de los 25 bloques.
- **Ruido y atípicos:** ruido gaussiano en las variables estandarizadas (σ = 0.1, 0.25 y 0.5) en entrenamiento y prueba; ruido en la etiqueta ordinal del entrenamiento (desplazar ±1 la nota del 5, 10 y 20 % de los vinos); atípicos en el entrenamiento (1 y 5 % de las filas con un valor multiplicado por 10).
- **Tamaño de muestra y escalabilidad:** curva de aprendizaje con el 10, 25, 50 y 100 % del entrenamiento; tiempo de ajuste al variar n y p.
- **Cambio de distribución:** entrenar con tintos y evaluar en blancos, y al revés (el tipo de vino sale de las variables).
- **Calibración:** curvas de confiabilidad, ECE (10 grupos por cuantiles), Brier y log-loss.
- **Incertidumbre:** conjuntos de predicción conformes por partición (75 % ajuste, 25 % calibración dentro del entrenamiento externo) con α = 0.10 y 0.05; cobertura empírica, tamaño medio y versión por clase (Mondrian).
- **Análisis de errores:** AUC por tipo de vino, por nota real y por tramos de alcohol; casos donde el BOF falla y el Random Forest no.

## 11. Cómputo

Equipo: AMD Ryzen 5 5600GT (6 núcleos físicos, 12 lógicos), 15.8 GB de RAM, Windows 11, Python 3.12.14, 11 procesos en paralelo con un hilo por modelo. Se reportan tiempos de búsqueda, de ajuste y de predicción, y el tamaño en memoria de cada modelo final.

## 12. Implementación

- Validación anidada: `experiments/correr_validacion.py` (módulos `src/validacion.py`, `src/benchmarks.py`, `src/optimizadores.py`, `src/data.py`).
- Ablación: `experiments/ablacion.py`. Pruebas: `src/pruebas.py`.
- Modelo original y sus pruebas unitarias: `src/original_model.py` y `tests/test_original_model.py`.
- Huellas SHA-256 del código al congelar el protocolo: anexo B.

## Anexo A. Rondas de diseño previas al protocolo

Todas con validación interna 3 × 3 repetida (semilla 7) sobre los 4256 vinos de entrenamiento del pliegue externo 1 (semilla 42). AUC medio de los 9 pliegues internos.

| Ronda | Qué se probó | Configuraciones | Resultado que orientó el diseño |
|---|---|---|---|
| 1 | Random Forest y Extra-Trees: evento binario, regresión sobre la nota y salida múltiple focalizada | 10 | Usar la nota completa sube el AUC (Extra-Trees sobre la nota 0.853 frente a Random Forest binario 0.847). |
| 2 | Extra-Trees de salida múltiple: ancho del núcleo de partición × ancho del núcleo de agregación × max_features | 39 | La ganancia viene sobre todo de agregar umbrales vecinos con un ancho intermedio (0.854), mejor que solo el umbral 7 (0.850) y que el peso uniforme (0.852). |
| 3 | Bootstrap, tamaño de hoja, centro y ancho del núcleo | 12 | El bootstrap completo no cambia el AUC (permite calibrar fuera de bolsa); hoja mínima 1; centro en el umbral del evento. |
| 4 | Implementación propia frente al prototipo (5 variantes de los núcleos) | 5 | La implementación en numpy reproduce el prototipo (0.8536 frente a 0.8536). |
| 5 | Barrido uno a uno de max_features, cortes por atributo, hoja mínima, desplazamiento y anchos | 14 | Raíz de p atributos, 1 corte, hoja 1 y ancho de agregación entre 0.8 y 1.5 desviaciones; con esto se fijaron los rangos de búsqueda. |

## Anexo B. Huellas SHA-256 del código al congelar el protocolo

| Archivo | SHA-256 |
|---|---|
| `src/original_model.py` | `b1ab9f08035de8aded1f5794912edeeaf8b65a6d6113206aac65eb6cf72cbfe0` |
| `src/data.py` | `06d7e07074e66b5861ea41005b507f4bdd00a16ae430d4b1dd47981d1b7a8269` |
| `src/benchmarks.py` | `9b23b14c841b41bd5b42da957c959656f9125bc0fd83b0f4a13c2e7e93dfade8` |
| `src/optimizadores.py` | `f2cf97212c9c48d3702704073f8448c60b4b70bc33b1f7417ad0883cd87fe4da` |
| `src/validacion.py` | `288a455c96a6c38ae40b11eb239083fd0b4b34ed358f40f69b88f7b74b5add17` |
| `src/pruebas.py` | `aab126f1067505751644df092a87663c378d961677b48d3951c3c332fe7614a6` |
| `experiments/correr_validacion.py` | `8333da5186e525f8549b461c05c5b9017cc7c316d82a64c45e457ee85d185ba8` |
| `experiments/ablacion.py` | `81f3d433c470c790f3b78cec8ed8b3b93277e7069cc9acae8da3720a680d811b` |
| `tests/test_original_model.py` | `be61e91f770b0bcce3222c717db84ed3159ce6a592949975768130cad77bdd5d` |
