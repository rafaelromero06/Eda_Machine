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

# Reproducibilidad y bibliografía

```{code-cell} ipython3
:tags: [remove-cell]
import sys, json, pathlib, platform
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import pandas as pd
from src.artefactos import version_instalada
from myst_nb import glue
from src.libro import mostrar, resultado
cv = resultado("cv_filas.csv")
glue("horas_cpu", f"{cv['tiempo_busqueda_s'].sum() / 3600:.1f}", display=False)
glue("n_tareas", len(cv), display=False)
```

## Entorno

```{code-cell} ipython3
:tags: [hide-input]
librerias = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "catboost", "optuna", "shap", "joblib",
             "dash", "dash-bootstrap-components", "plotly", "matplotlib", "pytest", "jupyter-book"]
mostrar(pd.DataFrame([{"Paquete": "python", "Versión": platform.python_version()}] + [{"Paquete": l, "Versión": version_instalada(l)} for l in librerias]))
```

Las versiones exactas están fijadas en `requirements.txt` (dashboard) y `requirements-dev.txt` (experimentos, pruebas y libro), y `.python-version` fija Python 3.12.14. El `Dockerfile` reproduce el entorno del dashboard. Cada `metadata.json` de `artifacts/` repite las versiones con que se entrenó el modelo.

**Equipo:** AMD Ryzen 5 5600GT (6 núcleos físicos, 12 lógicos), 15.8 GB de RAM, Windows 11. Los experimentos usaron 11 procesos en paralelo con un hilo por modelo.

## Lista de verificación de reproducibilidad

| Ítem | Dónde |
|---|---|
| Código público y versionado | https://github.com/rafaelromero06/Eda_Machine, carpeta `entregable3/` |
| Identificador persistente | PENDIENTE: DOI de Zenodo de la *release* final |
| Protocolo pre-registrado con fecha | `experiments/protocolo.md`, commit `8bc0c2d`, tag `protocolo-e3-v1` (26/10/2026) |
| Datos u obtención exacta | `data/` (vino y conjuntos adicionales, sin modificar) y `data/adicionales/procedencia.json` con los DOI de UCI |
| Semillas | particiones 42–46; estimadores 42; optimizador = semilla de la repetición + pliegue; robustez: semillas 0–9 |
| Rangos de hiperparámetros | `src/benchmarks.py::ESPACIOS` y capítulo 3 |
| Número de ejecuciones | {glue:text}`n_tareas` pliegues externos (10 modelos × 25 en vino y 10 × 5 en cada conjunto adicional), 40 evaluaciones de hiperparámetros cada uno |
| Medidas de dispersión | media ± desviación estándar en 25 bloques, intervalos bootstrap y BCa, cajas por semilla |
| Pruebas unitarias | `tests/` (pytest), ejecutadas en cada *push* por GitHub Actions |
| Tiempo de cómputo | {glue:text}`horas_cpu` horas de CPU solo en la búsqueda de la validación principal; ver capítulo 3 por modelo |
| Entorno fijado | `requirements*.txt`, `.python-version`, `Dockerfile` |
| Orden de ejecución | `README.md` y `experiments/ejecutar_todo.py` |

## Tabla maestra de experimentos

```{code-cell} ipython3
:tags: [hide-input]
maestra = resultado("tabla_maestra_e3.csv")
resumen = maestra.groupby(["origen", "conjunto", "tarea"]).size().rename("filas").reset_index()
mostrar(resumen.rename(columns={"origen": "Origen", "conjunto": "Conjunto", "tarea": "Tarea", "filas": "Combinaciones"}))
```

`experiments/resultados/tabla_maestra_e3.csv` reúne las 136 combinaciones del Entregable 2 (108 de clasificación y 28 de regresión, con 36 indicadores cada una) y las 50 filas del Entregable 3 (10 modelos × 5 conjuntos) con el esquema de validación, las repeticiones, las semillas, las métricas del bucle externo (media y desviación), las evaluaciones y los tiempos.

## Declaración de autoría y uso de herramientas

PENDIENTE: declaración firmada por el equipo (ver el capítulo 9).

## Bibliografía

```{bibliography}
```
