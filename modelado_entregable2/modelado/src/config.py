
from pathlib import Path


RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Validación cruzada anidada 
# ---------------------------------------------------------------------------

K_OUTER = 5


K_INNER = 3

# ---------------------------------------------------------------------------
# Presupuesto computacional de la optimización de hiperparámetros
# ---------------------------------------------------------------------------

PRESUPUESTO_EVALUACIONES = 40

GA_PROB_CRUCE = 0.6
GA_PROB_MUTACION = 0.3       # probabilidad de mutar un individuo
GA_PROB_MUTACION_GEN = 0.25  # prob. de mutar cada gen, dado que el individuo muta
GA_TAM_ELITE = 1             # número de individuos que pasan sin cambios (elitismo)
GA_TORNEO_K = 3              # tamaño del torneo en la selección
GA_N_ALELOS = 8              # valores discretos por hiperparámetro continuo


import os as _os
N_WORKERS = max(1, (_os.cpu_count() or 2) - 1)
N_JOBS = 1

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
RUTA_BASE = Path(__file__).resolve().parents[1]           # .../modelado
RUTA_DATA = RUTA_BASE / "data"
RUTA_EXPERIMENTOS = RUTA_BASE / "experiments"
RUTA_LOGS = RUTA_EXPERIMENTOS / "logs"
RUTA_FIGURAS = RUTA_EXPERIMENTOS / "figures"

RUTA_LOGS.mkdir(parents=True, exist_ok=True)


RUTA_LOG_CLASIFICACION = RUTA_LOGS / f"resultados_clasificacion_p{PRESUPUESTO_EVALUACIONES}.csv"
RUTA_LOG_REGRESION = RUTA_LOGS / f"resultados_regresion_p{PRESUPUESTO_EVALUACIONES}.csv"
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Definición de tareas y columnas
# ---------------------------------------------------------------------------
COLUMNAS_NUMERICAS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
COLUMNAS_CATEGORICAS = ["tipo_vino"]


UMBRAL_CLASIFICACION = 7
