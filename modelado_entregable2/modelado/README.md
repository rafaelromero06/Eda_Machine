# Modelado — Wine Quality (Entregable 2)

Comparación de 7 modelos × 4 técnicas de balanceo × 4 optimizadores de hiperparámetros (136 combinaciones) con validación cruzada anidada 5 × 3, para clasificar vinos buenos (`quality ≥ 7`) y predecir la nota de calidad.

## Entorno

Python 3.12.14 con las versiones fijadas en `requirements.txt` (o `environment.yml` para conda). La lista completa de paquetes del entorno usado, con el equipo en que se corrió, está en `experiments/entorno.txt`.

```
conda env create -f environment.yml      # o: pip install -r requirements.txt
conda activate mi_entorno
```

## Orden de ejecución (desde esta carpeta)

| Paso | Qué hace | Tiempo aprox. (6 núcleos, 11 procesos) |
|---|---|---|
| `python run_experimentos.py` | Validación anidada de las 136 combinaciones; guarda un fold por fila en `experiments/logs/` (se puede interrumpir y reanudar) | 1.5 h de reloj (16.6 h de CPU) |
| `python generar_predicciones.py` | Reconstruye las predicciones fuera de muestra de los 5,320 vinos reentrenando los hiperparámetros del log | 2 min |
| `notebooks/01_resultados_modelado.ipynb` | Comparación de modelos, balanceo y optimizadores; Friedman–Nemenyi; configuración recomendada | < 1 min |
| `notebooks/02_evaluacion_detallada.ipynb` | Curvas ROC y PR, matrices de confusión, umbral, calibración y recalibración, residuos de regresión | 1 min |
| `notebooks/03_pruebas_estadisticas.ipynb` | W de Kendall, Nemenyi por pares, DeLong, Diebold–Mariano, MCS, tamaños de efecto, IC bootstrap | 1 min |
| `notebooks/04_interpretabilidad.ipynb` | SHAP (importancia, dirección, dependencias) y LIME (casos, estabilidad, acuerdo con SHAP) | 6 min la primera vez |
| `notebooks/05_costo_computacional.ipynb` | Complejidad teórica, escalamiento empírico, perfilamiento, paralelismo y comparación estándar vs. optimizada | 10 min la primera vez |
| `notebooks/06_tabla_maestra.ipynb` | Tabla maestra con las 136 combinaciones y todos los indicadores | < 1 min |

Los notebooks 02 a 05 guardan sus cálculos pesados (predicciones recalibradas, valores SHAP y LIME, mediciones de tiempo) en `experiments/` y los reutilizan en las siguientes ejecuciones.

## Estructura

```
src/
  config.py, data_prep.py, model_factory.py, balance_factory.py, pipeline_builder.py,
  optimizers.py, nested_cv.py          # experimento (validación anidada en paralelo)
  predicciones.py                      # predicciones fuera de muestra, recalibración, umbral interno
  seleccion.py                         # resumen de logs, mejor por modelo, regla de 1 error estándar
  evaluacion.py                        # calibración (ECE, Brier, pendiente) y pruebas de residuos
  estadistica.py                       # Friedman/Nemenyi, DeLong, Diebold-Mariano, MCS, efectos, bootstrap
  interpretabilidad.py                 # SHAP fuera de muestra y LIME
  costo.py                             # complejidad, escalamiento, perfilamiento, planificación
  tabla_maestra.py, entorno.py, graficos.py
experiments/
  logs/            un fold externo por fila (resultado de run_experimentos.py)
  predicciones/    predicciones fuera de muestra y su verificación contra el log
  figures/         fig01–fig32
  tablas/          tabla01–tabla28 y tabla_maestra.csv (con su diccionario)
  interpretabilidad/, costo/   resultados intermedios de los notebooks 04 y 05
```
