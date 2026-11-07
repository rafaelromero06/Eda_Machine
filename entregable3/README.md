# Entregable 3 · Dashboard predictivo, modelo original y evaluación rigurosa

Proyecto final de Machine Learning (Universidad del Norte) sobre la calidad del vino Vinho Verde.
Autores: Roger Daza, Matteo Del Sole y Rafael Romero.

- **Tarea:** clasificar si un vino es bueno (nota de calidad ≥ 7) a partir de 11 variables fisicoquímicas y el tipo de vino.
- **Modelo original:** Bosque Ordinal Focalizado (`src/original_model.py`), programado desde cero con numpy y scipy.
- **Protocolo pre-registrado:** `experiments/protocolo.md`, congelado con la etiqueta de Git `protocolo-e3-v1` (26 de octubre de 2026).
- **Dashboard:** «Cata Ciega», aplicación Dash multipágina (`app.py`, `pages/`) con las secciones Contexto, EDA y Modelos, un juego de cata a ciegas y la ficha del modelo, desplegable en Render.
- **Informe:** Jupyter Book en `book/`.

## Estructura

```
entregable3/
|-- app.py                     punto de entrada del dashboard (expone server para gunicorn)
|-- pages/                     contexto.py, eda.py, modelos.py, cata.py, ficha.py
|-- src/
|   |-- original_model.py      Bosque Ordinal Focalizado
|   |-- data.py                carga de datos, esquema de entrada y caché del dashboard
|   |-- plots.py               figuras Plotly del dashboard
|   |-- prediccion.py          validación de entradas, predicción, vinos gemelos, palancas y lotes
|   |-- tablero.py             piezas comunes del dashboard (encabezados, tarjetas, glosario)
|   |-- tablero_modelos.py     contenido de las pestañas de Modelos
|   |-- benchmarks.py          los 7 modelos del Entregable 2, LightGBM, CatBoost y el modelo original
|   |-- optimizadores.py       Grid Search adaptativo, Random Search y bayesiano (TPE)
|   |-- validacion.py          validación anidada repetida con predicciones fuera de muestra
|   |-- pruebas.py             Friedman, Nemenyi, Wilcoxon, t corregida, DeLong, bootstrap BCa, Cliff, pruebas bayesianas
|   |-- analisis.py            tablas de resultados y veredicto R3
|   |-- robustez.py            sensibilidad, semillas, ruido, tamaño, cambio de distribución, conformes
|   |-- artefactos.py          reentrenamiento final, serialización y metadata.json
|   |-- explicaciones.py       SHAP local para cada modelo final (incluido el árbol propio)
|   |-- auditoria.py           auditoría de fuga de información
|   |-- tablero_original.py    contenido de la pestaña del modelo original
|   `-- libro.py               estilo y utilidades de las figuras del libro
|-- experiments/
|   |-- protocolo.md           protocolo pre-registrado
|   |-- correr_validacion.py   experimentos principales (vino 5 x 5 y conjuntos adicionales)
|   |-- ablacion.py            estudio de ablación
|   |-- modelos_finales.py     búsqueda final, reentrenamiento y artefactos
|   |-- robustez.py            estudios de robustez
|   |-- memoria.py             memoria del ajuste de cada modelo final
|   |-- analisis.py            pruebas de hipótesis y tablas
|   |-- tabla_maestra_e3.py    tabla maestra de experimentos
|   |-- documentos.py          ficha del modelo (MODEL_CARD.md)
|   |-- preparar_app.py        datos precalculados del dashboard
|   |-- ejecutar_todo.py       todos los pasos en orden
|   |-- bitacora_diseno.csv    configuraciones evaluadas en los pilotos
|   |-- revision_literatura/   búsqueda bibliográfica, citas, repositorios y matriz de novedad
|   |-- pilotos/               rondas de diseño previas al protocolo
|   `-- resultados/            CSV de resultados y tablas
|-- artifacts/                 modelos finales serializados con metadata.json
|-- data/                      datos (vino y conjuntos adicionales), DATA_STATEMENT.md y datos del dashboard
|-- assets/                    estilos, animaciones (cata.js) y azulejo del dashboard
|-- tests/                     pruebas con pytest
|-- book/                      fuentes del Jupyter Book
|-- MODEL_CARD.md              ficha del modelo original
|-- CITATION.cff, LICENSE, Dockerfile, requirements.txt, requirements-dev.txt
```

## Instalación

Python 3.12. Para usar el dashboard basta con `requirements.txt`; para reproducir los experimentos y compilar el libro se necesita `requirements-dev.txt`.

`requirements.txt` instala `xgboost-cpu==3.4.1`, la misma versión de XGBoost sin soporte de GPU, porque su rueda para Linux pesa 5.5 MB en lugar de 55 MB y el servidor de Render tiene poca memoria. Los modelos se cargan igual con cualquiera de las dos; si el entorno ya tiene `xgboost==3.4.1`, no hace falta instalar también `xgboost-cpu`.

```bash
pip install -r requirements-dev.txt
```

## Uso

Dashboard en local (http://127.0.0.1:8050):

```bash
python app.py
```

Pruebas:

```bash
pytest
```

Reproducir todo, en orden (los tiempos son de un equipo de 6 núcleos con 11 procesos):

```bash
python experiments/correr_validacion.py todo    # validación anidada: vino 5 x 5 y 4 conjuntos adicionales (varias horas)
python experiments/ablacion.py                  # ablación del modelo original
python experiments/modelos_finales.py           # búsqueda final, reentrenamiento y artefactos
python experiments/robustez.py                  # estudios de robustez (en paralelo y escalabilidad)
python experiments/memoria.py                   # memoria del ajuste de cada modelo final
python experiments/analisis.py                  # pruebas de hipótesis, tablas y veredicto
python experiments/tabla_maestra_e3.py          # tabla maestra
python experiments/documentos.py                # ficha del modelo (MODEL_CARD.md)
python experiments/preparar_app.py              # datos precalculados del dashboard
jupyter-book build book/                        # informe
```

`python experiments/ejecutar_todo.py` corre todos los pasos de experimentos en orden (acepta el nombre del paso desde el que se quiere empezar, por ejemplo `ablacion`). Las corridas se pueden interrumpir y retomar: `correr_validacion.py` salta las tareas que ya están en `experiments/resultados/cv_filas.csv`.

## Despliegue en Render

El archivo `render.yaml` de la raíz del repositorio define el servicio (carpeta raíz `entregable3`, `pip install -r requirements.txt` y `gunicorn app:server`). En Render: **New → Blueprint**, conectar el repositorio de GitHub y aceptar. También se puede crear un *Web Service* a mano con *Root Directory* `entregable3`, *Build Command* `pip install -r requirements.txt` y *Start Command* `gunicorn app:server --workers 1 --threads 4 --timeout 120`. El plan gratuito se suspende sin tráfico: conviene abrir la URL unos minutos antes de la sustentación.

## Licencias y cita

Código bajo licencia MIT (`LICENSE`). Datos de UCI bajo CC BY 4.0 (ver `data/DATA_STATEMENT.md`). Para citar el repositorio, ver `CITATION.cff`.
