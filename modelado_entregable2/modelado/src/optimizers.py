
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

import optuna
from deap import base, creator, tools

from . import config
from .config import (
    GA_PROB_CRUCE, GA_PROB_MUTACION, GA_PROB_MUTACION_GEN,
    GA_TAM_ELITE, GA_TORNEO_K, GA_N_ALELOS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ResultadoOptimizacion:
    mejor_estimador: object
    mejores_hiperparametros: dict
    mejor_score_interno: float
    historia: list            # mejor score visto hasta la evaluación i (anytime performance)
    n_evaluaciones: int
    tiempo_segundos: float
    extra: dict = field(default_factory=dict)


# =============================================================================
# Adaptadores: espacio canónico -> formato específico de cada método
# =============================================================================
def _valores_continuos(spec: dict, n_puntos: int) -> list:
    low, high = spec["low"], spec["high"]
    vals = np.geomspace(low, high, n_puntos) if spec["log"] else np.linspace(low, high, n_puntos)
    if spec["type"] == "int":
        vals = sorted({int(round(v)) for v in vals})
    else:
        # redondeo a 6 cifras SIGNIFICATIVAS (no decimales): con round(v, 8)
        # valores como var_smoothing=1e-11 colapsaban a 0.0 y la grilla
        # quedaba con puntos repetidos y fuera del rango declarado
        vals = sorted({float(f"{float(v):.6g}") for v in vals})
    return vals


def _n_puntos_adaptativo(space: dict, presupuesto: int) -> int:
    """Cuántos puntos usar por hiperparámetro continuo para que el tamaño
    TOTAL de la grilla (incluyendo el producto de las cardinalidades
    categóricas, que no se pueden subdividir) quede en el orden de
    `presupuesto`, en vez de crecer sin control con la dimensionalidad
    (p. ej. Random Forest tiene 4 hiperparámetros: con 3 puntos fijos por
    cada uno la grilla da 3x3x3x3=81 evaluaciones, ~5x el presupuesto de
    los otros 3 métodos). Esto es justamente la explosión combinatoria de
    la Sección 2 -- aquí se controla para que la comparación de métodos en
    la Sección 3.4 siga siendo razonable, documentando el tamaño real de
    cada grilla en los resultados en vez de forzarlo a calzar exacto."""
    continuos = [s for s in space.values() if s["type"] != "categorical"]
    mult_categorico = 1
    for s in space.values():
        if s["type"] == "categorical":
            mult_categorico *= len(s["categories"])
    if not continuos:
        return 2
    presupuesto_continuo = max(presupuesto / mult_categorico, 2 * len(continuos))
    return max(2, round(presupuesto_continuo ** (1 / len(continuos))))


def espacio_a_grid(space: dict, presupuesto: int = 16) -> dict:
    n_puntos = _n_puntos_adaptativo(space, presupuesto)
    return {
        nombre: (list(spec["categories"]) if spec["type"] == "categorical"
                 else _valores_continuos(spec, n_puntos))
        for nombre, spec in space.items()
    }


class _LogUniformInt:
    """Distribución compatible con RandomizedSearchCV (método .rvs) para un
    entero muestreado en escala logarítmica."""

    def __init__(self, low, high):
        self._d = loguniform(low, high)

    def rvs(self, random_state=None, size=None):
        vals = self._d.rvs(random_state=random_state, size=size if size else 1)
        vals = np.round(vals).astype(int)
        return vals if size else int(vals[0])


def espacio_a_distribuciones(space: dict) -> dict:
    dist = {}
    for nombre, spec in space.items():
        if spec["type"] == "categorical":
            dist[nombre] = list(spec["categories"])
        elif spec["type"] == "float":
            dist[nombre] = (loguniform(spec["low"], spec["high"]) if spec["log"]
                             else uniform(spec["low"], spec["high"] - spec["low"]))
        else:  # int
            dist[nombre] = (_LogUniformInt(spec["low"], spec["high"]) if spec["log"]
                             else randint(spec["low"], spec["high"] + 1))
    return dist


def espacio_a_optuna(trial, space: dict) -> dict:
    params = {}
    for nombre, spec in space.items():
        if spec["type"] == "categorical":
            params[nombre] = trial.suggest_categorical(nombre, spec["categories"])
        elif spec["type"] == "float":
            params[nombre] = trial.suggest_float(nombre, spec["low"], spec["high"], log=spec["log"])
        else:
            params[nombre] = trial.suggest_int(nombre, spec["low"], spec["high"], log=spec["log"])
    return params


def espacio_a_genetico(space: dict, n_alelos: int = 6) -> dict:
    return {
        nombre: (list(spec["categories"]) if spec["type"] == "categorical"
                 else _valores_continuos(spec, n_alelos))
        for nombre, spec in space.items()
    }


def _prefijar(d: dict) -> dict:
    return {f"modelo__{k}": v for k, v in d.items()}


# =============================================================================
# 1 GRID SEARCH
# =============================================================================
def buscar_grid(pipe, space, X, y, cv_inner, scoring, seed, presupuesto=16, n_jobs=config.N_JOBS) -> ResultadoOptimizacion:
    t0 = time.time()
    grid = espacio_a_grid(space, presupuesto=presupuesto)
    gs = GridSearchCV(pipe, _prefijar(grid), cv=cv_inner, scoring=scoring,
                       n_jobs=n_jobs, refit=True, error_score=np.nan)
    gs.fit(X, y)
    tiempo = time.time() - t0

    scores = np.nan_to_num(gs.cv_results_["mean_test_score"], nan=-np.inf)
    historia = list(np.maximum.accumulate(scores))
    mejores = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}

    return ResultadoOptimizacion(
        mejor_estimador=gs.best_estimator_, mejores_hiperparametros=mejores,
        mejor_score_interno=gs.best_score_, historia=historia,
        n_evaluaciones=len(scores), tiempo_segundos=tiempo,
        extra={"tamano_grilla": len(scores), "metodo": "Grid Search"},
    )


# =============================================================================
# 2 RANDOM SEARCH
# =============================================================================
def buscar_random(pipe, space, X, y, cv_inner, scoring, seed, presupuesto=16, n_jobs=config.N_JOBS) -> ResultadoOptimizacion:
    t0 = time.time()
    dist = espacio_a_distribuciones(space)
    rs = RandomizedSearchCV(pipe, _prefijar(dist), n_iter=presupuesto, cv=cv_inner,
                             scoring=scoring, n_jobs=n_jobs, random_state=seed,
                             refit=True, error_score=np.nan)
    rs.fit(X, y)
    tiempo = time.time() - t0

    scores = np.nan_to_num(rs.cv_results_["mean_test_score"], nan=-np.inf)
    historia = list(np.maximum.accumulate(scores))
    mejores = {k.replace("modelo__", ""): v for k, v in rs.best_params_.items()}

    return ResultadoOptimizacion(
        mejor_estimador=rs.best_estimator_, mejores_hiperparametros=mejores,
        mejor_score_interno=rs.best_score_, historia=historia,
        n_evaluaciones=len(scores), tiempo_segundos=tiempo,
        extra={"metodo": "Random Search"},
    )


# =============================================================================
# 3 OPTIMIZACIÓN BAYESIANA 
# =============================================================================
def buscar_bayesiano(pipe, space, X, y, cv_inner, scoring, seed, presupuesto=16, n_jobs=config.N_JOBS) -> ResultadoOptimizacion:
    t0 = time.time()

    def objetivo(trial):
        params = espacio_a_optuna(trial, space)
        modelo = clone(pipe).set_params(**_prefijar(params))
        scores = cross_val_score(modelo, X, y, cv=cv_inner, scoring=scoring,
                                  n_jobs=n_jobs, error_score=np.nan)
        val = np.nanmean(scores)
        return val if not np.isnan(val) else -np.inf

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objetivo, n_trials=presupuesto, n_jobs=1, show_progress_bar=False)
    tiempo = time.time() - t0

    valores = [t.value for t in study.trials]
    historia = list(np.maximum.accumulate(valores))
    mejores = study.best_trial.params

    mejor_pipe = clone(pipe).set_params(**_prefijar(mejores))
    mejor_pipe.fit(X, y)

    return ResultadoOptimizacion(
        mejor_estimador=mejor_pipe, mejores_hiperparametros=mejores,
        mejor_score_interno=study.best_value, historia=historia,
        n_evaluaciones=len(study.trials), tiempo_segundos=tiempo,
        extra={
            "metodo": "Bayesiano (Optuna)",
            "surrogate_model": "TPE (Tree-structured Parzen Estimator): modela p(x|y<umbral)=l(x) "
                                "y p(x|y>=umbral)=g(x) por separado, en vez de un único proceso "
                                "gaussiano sobre f(x).",
            "funcion_adquisicion": "Expected Improvement, aproximada como la razón l(x)/g(x) "
                                    "(a mayor razón, mayor mejora esperada); TPE la maximiza "
                                    "muestreando de l(x). Favorece explotación de regiones "
                                    "'buenas' ya observadas sin dejar de muestrear g(x).",
        },
    )


# =============================================================================
# 4) ALGORITMO GENÉTICO 
# =============================================================================
def _diversidad_poblacion(poblacion) -> float:
    """Entropía de Shannon normalizada, promediada por gen, como indicador
    de diversidad genética (1 = máxima dispersión, 0 = población idéntica)."""
    arr = np.array(poblacion)
    entropias = []
    for col in arr.T:
        _, counts = np.unique(col, return_counts=True)
        p = counts / counts.sum()
        ent = -(p * np.log(p + 1e-12)).sum()
        ent_max = np.log(len(counts)) if len(counts) > 1 else 1.0
        entropias.append(ent / ent_max if ent_max > 0 else 0.0)
    return float(np.mean(entropias))


def buscar_genetico(pipe, space, X, y, cv_inner, scoring, seed, presupuesto=16, n_jobs=config.N_JOBS) -> ResultadoOptimizacion:
    t0 = time.time()
    random.seed(seed)  # semilla global: DEAP usa random.* internamente, no acepta un RNG propio

    # Discretización de los genes continuos: se parte de GA_N_ALELOS valores
    # por gen y se duplica mientras el espacio discreto total sea más chico
    # que 2x el presupuesto. Sin esto, en modelos de un solo hiperparámetro
    # (Naive Bayes, Ridge, Lasso) el GA solo tendría 8 configuraciones
    # posibles y no podría gastar un presupuesto de 40 evaluaciones.
    n_alelos = GA_N_ALELOS
    genes = espacio_a_genetico(space, n_alelos=n_alelos)
    hay_continuos = any(spec["type"] != "categorical" for spec in space.values())
    while (hay_continuos and n_alelos < 128
           and int(np.prod([len(v) for v in genes.values()])) < 2 * presupuesto):
        n_alelos *= 2
        genes = espacio_a_genetico(space, n_alelos=n_alelos)
    nombres = list(genes.keys())
    tamanos = [len(genes[n]) for n in nombres]
    n_configuraciones_posibles = int(np.prod(tamanos))


    poblacion_n = max(4, round(math.sqrt(presupuesto)))
    max_generaciones = 4 * max(2, round(presupuesto / poblacion_n))
    presupuesto_real = min(presupuesto, n_configuraciones_posibles)

    cache: dict[tuple, float] = {}
    historial_evals: list[float] = []

    def decodificar(ind):
        return {n: genes[n][i] for n, i in zip(nombres, ind)}

    def evaluar(ind):
        clave = tuple(ind)
        if clave in cache:
            return (cache[clave],)
        if len(historial_evals) >= presupuesto_real:
            # presupuesto agotado: el individuo no se evalúa y queda con el
            # peor fitness posible (no puede desplazar a la élite)
            return (-np.inf,)
        params = decodificar(ind)
        modelo = clone(pipe).set_params(**_prefijar(params))
        scores = cross_val_score(modelo, X, y, cv=cv_inner, scoring=scoring,
                                  n_jobs=n_jobs, error_score=np.nan)
        fitness = float(np.nanmean(scores))
        if np.isnan(fitness):
            fitness = -np.inf
        cache[clave] = fitness
        historial_evals.append(fitness)
        return (fitness,)

    if not hasattr(creator, "FitnessMaxWQ"):
        creator.create("FitnessMaxWQ", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualWQ"):
        creator.create("IndividualWQ", list, fitness=creator.FitnessMaxWQ)

    toolbox = base.Toolbox()
    toolbox.register("individuo", lambda: creator.IndividualWQ(random.randrange(t) for t in tamanos))
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("select", tools.selTournament, tournsize=GA_TORNEO_K)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    def _mutar(ind):
        for i in range(len(ind)):
            if random.random() < GA_PROB_MUTACION_GEN:
                ind[i] = random.randrange(tamanos[i])
        return (ind,)

    toolbox.register("mutate", _mutar)

    pop = toolbox.poblacion(n=poblacion_n)
    for ind in pop:
        ind.fitness.values = evaluar(ind)

    diversidad_por_gen = [_diversidad_poblacion(pop)]

    generacion = 0
    while len(historial_evals) < presupuesto_real and generacion < max_generaciones:
        generacion += 1
        elite = [copy.deepcopy(e) for e in tools.selBest(pop, GA_TAM_ELITE)]

        descendencia = [copy.deepcopy(d) for d in toolbox.select(pop, len(pop) - GA_TAM_ELITE)]
        for h1, h2 in zip(descendencia[::2], descendencia[1::2]):
            if random.random() < GA_PROB_CRUCE:
                toolbox.mate(h1, h2)
                del h1.fitness.values
                del h2.fitness.values
        for mut in descendencia:
            if random.random() < GA_PROB_MUTACION:
                toolbox.mutate(mut)
                del mut.fitness.values

        # Eliminación de duplicados: si un hijo es idéntico a una
        # configuración ya evaluada, se le fuerza a cambiar un gen al azar
        # (hasta 10 intentos). Sin esto, con poblaciones pequeñas la
        # población converge rápido y las generaciones siguientes solo
        # producen copias ya evaluadas -> el GA deja de explorar y no llega
        # a gastar su presupuesto de evaluaciones.
        for ind in descendencia:
            intentos = 0
            while (tuple(ind) in cache and intentos < 10
                   and len(cache) < n_configuraciones_posibles):
                gen = random.randrange(len(ind))
                ind[gen] = random.randrange(tamanos[gen])
                intentos += 1
                if ind.fitness.valid:
                    del ind.fitness.values
            if not ind.fitness.valid:
                ind.fitness.values = evaluar(ind)

        pop = elite + descendencia
        diversidad_por_gen.append(_diversidad_poblacion(pop))

    tiempo = time.time() - t0
    mejor = tools.selBest(pop, 1)[0]
    mejores = decodificar(mejor)
    historia = list(np.maximum.accumulate(historial_evals))

    mejor_pipe = clone(pipe).set_params(**_prefijar(mejores))
    mejor_pipe.fit(X, y)

    return ResultadoOptimizacion(
        mejor_estimador=mejor_pipe, mejores_hiperparametros=mejores,
        mejor_score_interno=mejor.fitness.values[0], historia=historia,
        n_evaluaciones=len(historial_evals), tiempo_segundos=tiempo,
        extra={
            "metodo": "Genético (DEAP)",
            "seleccion": f"Torneo, tournsize={GA_TORNEO_K}",
            "cruce": f"Uniforme (cxUniform, indpb=0.5), prob.={GA_PROB_CRUCE}",
            "mutacion": f"Reinicio aleatorio de gen, prob. individuo={GA_PROB_MUTACION}, "
                        f"prob. por gen={GA_PROB_MUTACION_GEN}",
            "elitismo": f"{GA_TAM_ELITE} individuo(s)/generación",
            "poblacion": poblacion_n, "generaciones": generacion,
            "diversidad_por_generacion": diversidad_por_gen,
        },
    )


OPTIMIZADORES = {
    "Grid Search": buscar_grid,
    "Random Search": buscar_random,
    "Bayesiano": buscar_bayesiano,
    "Genético": buscar_genetico,
}
