from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import optuna
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ResultadoOptimizacion:
    mejor_estimador: object
    mejores_hiperparametros: dict
    mejor_score_interno: float
    historia: list
    n_evaluaciones: int
    tiempo_segundos: float
    tiempo_refit: float
    extra: dict = field(default_factory=dict)


def _valores_continuos(spec: dict, n_puntos: int) -> list:
    low, high = spec["low"], spec["high"]
    vals = np.geomspace(low, high, n_puntos) if spec["log"] else np.linspace(low, high, n_puntos)
    if spec["type"] == "int":
        return sorted({int(round(v)) for v in vals})
    return sorted({float(f"{float(v):.6g}") for v in vals})


def _n_puntos_adaptativo(space: dict, presupuesto: int) -> int:
    continuos = [s for s in space.values() if s["type"] != "categorical"]
    mult_categorico = 1
    for s in space.values():
        if s["type"] == "categorical":
            mult_categorico *= len(s["categories"])
    if not continuos:
        return 2
    presupuesto_continuo = max(presupuesto / mult_categorico, 2 * len(continuos))
    return max(2, round(presupuesto_continuo ** (1 / len(continuos))))


def espacio_a_grid(space: dict, presupuesto: int) -> dict:
    n_puntos = _n_puntos_adaptativo(space, presupuesto)
    return {nombre: (list(spec["categories"]) if spec["type"] == "categorical" else _valores_continuos(spec, n_puntos))
            for nombre, spec in space.items()}


class _LogUniformInt:
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
        else:
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


def _prefijar(d: dict) -> dict:
    return {f"modelo__{k}": v for k, v in d.items()}


def buscar_grid(pipe, space, X, y, cv, scoring, seed, presupuesto=40, encolar=None) -> ResultadoOptimizacion:
    """Grid Search con grilla adaptativa (tamaño cercano al presupuesto), igual al Entregable 2."""
    t0 = time.time()
    gs = GridSearchCV(pipe, _prefijar(espacio_a_grid(space, presupuesto)), cv=cv, scoring=scoring,
                      n_jobs=1, refit=True, error_score=np.nan)
    gs.fit(X, y)
    scores = np.nan_to_num(gs.cv_results_["mean_test_score"], nan=-np.inf)
    return ResultadoOptimizacion(
        mejor_estimador=gs.best_estimator_,
        mejores_hiperparametros={k.replace("modelo__", ""): v for k, v in gs.best_params_.items()},
        mejor_score_interno=float(gs.best_score_), historia=list(np.maximum.accumulate(scores)),
        n_evaluaciones=len(scores), tiempo_segundos=time.time() - t0, tiempo_refit=float(gs.refit_time_),
        extra={"metodo": "Grid Search"})


def buscar_random(pipe, space, X, y, cv, scoring, seed, presupuesto=40, encolar=None) -> ResultadoOptimizacion:
    """Random Search con las distribuciones del espacio canónico."""
    t0 = time.time()
    rs = RandomizedSearchCV(pipe, _prefijar(espacio_a_distribuciones(space)), n_iter=presupuesto, cv=cv,
                            scoring=scoring, n_jobs=1, random_state=seed, refit=True, error_score=np.nan)
    rs.fit(X, y)
    scores = np.nan_to_num(rs.cv_results_["mean_test_score"], nan=-np.inf)
    return ResultadoOptimizacion(
        mejor_estimador=rs.best_estimator_,
        mejores_hiperparametros={k.replace("modelo__", ""): v for k, v in rs.best_params_.items()},
        mejor_score_interno=float(rs.best_score_), historia=list(np.maximum.accumulate(scores)),
        n_evaluaciones=len(scores), tiempo_segundos=time.time() - t0, tiempo_refit=float(rs.refit_time_),
        extra={"metodo": "Random Search"})


def buscar_bayesiano(pipe, space, X, y, cv, scoring, seed, presupuesto=40, encolar=None) -> ResultadoOptimizacion:
    """Optimización bayesiana con TPE (Optuna); ``encolar`` agrega configuraciones iniciales."""
    t0 = time.time()

    def objetivo(trial):
        modelo = clone(pipe).set_params(**_prefijar(espacio_a_optuna(trial, space)))
        val = np.nanmean(cross_val_score(modelo, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan))
        return val if not np.isnan(val) else -np.inf

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    for params in encolar or []:
        study.enqueue_trial(params)
    study.optimize(objetivo, n_trials=presupuesto, n_jobs=1, show_progress_bar=False)
    mejores = study.best_trial.params
    mejor_pipe = clone(pipe).set_params(**_prefijar(mejores))
    t1 = time.time()
    mejor_pipe.fit(X, y)
    tiempo_refit = time.time() - t1
    return ResultadoOptimizacion(
        mejor_estimador=mejor_pipe, mejores_hiperparametros=mejores, mejor_score_interno=float(study.best_value),
        historia=list(np.maximum.accumulate([t.value for t in study.trials])), n_evaluaciones=len(study.trials),
        tiempo_segundos=time.time() - t0, tiempo_refit=tiempo_refit, extra={"metodo": "Bayesiano (TPE)"})


OPTIMIZADORES = {"Grid Search": buscar_grid, "Random Search": buscar_random, "Bayesiano": buscar_bayesiano}
