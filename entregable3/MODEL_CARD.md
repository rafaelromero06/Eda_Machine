# Ficha del modelo: Bosque Ordinal Focalizado

## Detalles del modelo

- **Desarrolladores:** Roger Daza, Matteo Del Sole y Rafael Romero (Universidad del Norte), curso de Machine Learning, 2026.
- **Versión:** 1.0 (2026-11-07). Artefacto: `artifacts/bosque_ordinal_focalizado/modelo.joblib` (SHA-256 `393b8457d4167611…`).
- **Tipo:** clasificador binario basado en un bosque de árboles extremadamente aleatorizados de salida múltiple, programado desde cero con numpy y scipy (`src/original_model.py`).
- **Algoritmo:** aprende las probabilidades acumuladas P(nota ≥ t) con una impureza ponderada por un núcleo centrado en el umbral 7, puntúa combinando los umbrales vecinos con un segundo núcleo y calibra el puntaje con los casos fuera de bolsa.
- **Hiperparámetros finales:** `n_estimators` = 438, `max_features` = None, `min_samples_leaf` = 1, `ancho_agregacion` = 1.281, `ancho_particion` = 2.55, `desplazamiento` = 0.3374. Se despliegan 231 árboles (el menor número con AUC fuera de bolsa a menos de 0.001 del bosque completo).
- **Licencia:** MIT. **Cita:** ver `CITATION.cff`.
- **Contacto:** repositorio https://github.com/rafaelromero06/Eda_Machine.

## Uso previsto

- **Uso principal:** estimar la probabilidad de que un vino Vinho Verde reciba una nota sensorial de 7 o más a partir de 11 mediciones fisicoquímicas de laboratorio y del tipo de vino, como apoyo para priorizar lotes antes de la cata.
- **Usuarios previstos:** enólogos, laboratorios de control de calidad y estudiantes que quieran entender la relación entre química y calidad percibida.
- **Usos no recomendados:** reemplazar la evaluación sensorial o la certificación oficial; fijar precios o rechazar lotes sin validación local; aplicarlo a vinos de otras regiones, uvas o épocas sin reentrenar; cualquier recomendación de consumo de alcohol o de salud.

## Factores

- **Tipo de vino:** los blancos son casi tres veces más que los tintos y tienen más vinos buenos; el desempeño se reporta por tipo.
- **Región y época:** solo vinos Vinho Verde de 2004 a 2007.
- **Etiqueta:** la nota es la mediana de al menos tres catadores; es subjetiva y discreta.

## Métricas

- **Principal:** AUC-ROC del evento en el bucle externo de la validación anidada 5 × 3 repetida 5 veces (25 particiones).
- **Secundarias:** PR-AUC, puntaje de Brier, log-loss, ECE, F1 con umbral 0.5 y cobertura de los conjuntos conformes.
- **Variación:** media ± desviación estándar en las 25 particiones e intervalo bootstrap del 95 % sobre las predicciones fuera de muestra.

## Datos de evaluación y de entrenamiento

- **Entrenamiento del artefacto:** los 5320 vinos sin duplicados del conjunto Wine Quality (UCI, CC BY 4.0), con la nota completa como respuesta.
- **Evaluación:** las mismas 25 particiones para todos los modelos; además cuatro conjuntos públicos adicionales (Abalone, Cardiotocografía, Student Performance y Heart Disease) elegidos antes de evaluar con el criterio del protocolo.
- **Preprocesamiento:** escalado estándar y one-hot del tipo de vino dentro de cada pliegue.

## Resultados cuantitativos

| Métrica | Bosque Ordinal Focalizado | Mejor benchmark (CatBoost) |
|---|---|---|
| AUC (media ± DE, 25 particiones) | 0.8649 ± 0.0123 | 0.8631 ± 0.0125 |
| AUC fuera de muestra, repetición 1 (IC 95 %) | 0.8637 (0.8520 a 0.8751) | 0.8488 (0.8371 a 0.8604) |
| PR-AUC | 0.6032 | 0.5991 |
| Brier | 0.1075 | 0.1152 |
| ECE | 0.0134 | 0.0424 |
| Cobertura del conjunto conforme al 90 % | 0.902 | 0.902 |

**Veredicto de superioridad (R3):** el modelo es **no superior** según la regla pre-registrada (el detalle de las pruebas está en el capítulo 6 del informe). Diferencia de AUC media frente al mejor benchmark: +0.0018; δ de Cliff = 0.09.

**Conjuntos adicionales (AUC media de 5 pliegues):**

| Conjunto | BOF | Mejor benchmark | Puesto del BOF |
|---|---|---|---|
| Abalone (edad por anillos) | 0.854 | SVM (0.864) | 5 de 10 |
| Cardiotocografía (estado fetal) | 0.987 | LightGBM (0.989) | 3 de 10 |
| Student Performance (portugués) | 0.841 | Random Forest (0.844) | 2 de 10 |
| Heart Disease (Cleveland) | 0.905 | Regresión Logística (L1/L2) (0.914) | 5 de 10 |

## Consideraciones éticas

- La nota sensorial refleja las preferencias de un panel concreto; el modelo reproduce esas preferencias, no una calidad objetiva.
- Un error del modelo puede descartar un lote bueno o promover uno mediocre: debe usarse como apoyo y no como decisión automática.
- Los conjuntos adicionales de salud y educación se usaron solo para comparar algoritmos; el modelo no está validado para decisiones sobre personas.

## Advertencias y recomendaciones

- Con pliegues formados por bloques de 10 vinos consecutivos del archivo (posibles lotes), el AUC baja de 0.859 a 0.853: los vinos de un mismo lote comparten información, así que con lotes nuevos el desempeño puede ser algo menor al reportado.
- La probabilidad calibrada es confiable dentro del rango de los datos de entrenamiento; fuera de los rangos válidos del esquema el formulario del dashboard rechaza la entrada.
- Para usar el modelo en otra región o época, se debe reentrenar y volver a validar con el mismo protocolo.
