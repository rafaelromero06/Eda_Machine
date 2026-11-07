from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from .benchmarks import ESPACIOS, MEJORES, construir_pipeline, da_probabilidades, puntaje, usa_respuesta_ordinal
from .data import cargar_conjunto
from .optimizadores import OPTIMIZADORES

SEMILLAS_REPETICION = [42, 43, 44, 45, 46]
K_EXTERNO, K_INTERNO, PRESUPUESTO = 5, 3, 40
PESO_MODELO = {"Random Forest": 10, "XGBoost": 8, "CatBoost": 9, "Bosque Ordinal Focalizado": 9, "SVM": 3,
               "LightGBM": 3, "KNN": 1.5}


def particiones(y_evento, semilla, k=K_EXTERNO):
    """Pliegues externos estratificados por el evento (idénticos para todos los modelos)."""
    return list(StratifiedKFold(k, shuffle=True, random_state=semilla).split(np.zeros(len(y_evento)), y_evento))


class PuntuadorEvento:
    """AUC del evento y >= umbral; permite optimizar modelos que se entrenan con la respuesta ordinal."""

    def __init__(self, umbral):
        self.umbral = umbral

    def __call__(self, estimador, X, y):
        return roc_auc_score(np.asarray(y) >= self.umbral, puntaje(estimador, X))


def metricas(y, s, probabilistico, pred=None):
    """Métricas del pliegue externo: AUC, PR-AUC y, si hay probabilidades, Brier, log-loss y umbral 0.5."""
    fila = {"auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
    if probabilistico:
        p = np.clip(s, 1e-12, 1 - 1e-12)
        fila.update({"brier": brier_score_loss(y, s), "log_loss": log_loss(y, p, labels=[0, 1])})
    else:
        fila.update({"brier": np.nan, "log_loss": np.nan})
    pred = (s >= 0.5).astype(int) if pred is None else pred
    fila.update({"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
                 "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)})
    return fila


def ejecutar_tarea(conjunto, modelo, repeticion, fold, presupuesto=PRESUPUESTO):
    """Un pliegue externo de la validación anidada: búsqueda interna, reajuste y evaluación.

    Returns
    -------
    fila : dict con métricas, tiempos e hiperparámetros.
    pred : DataFrame con el índice de cada caso de prueba y su puntaje.
    """
    warnings.filterwarnings("ignore")
    datos = cargar_conjunto(conjunto)
    semilla = SEMILLAS_REPETICION[repeticion]
    X, y_ord, y_ev = datos["X"], datos["y_ord"], datos["y_evento"]
    tr, te = particiones(y_ev, semilla)[fold]
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    internas = particiones(y_ev[tr], semilla, K_INTERNO)
    ordinal = usa_respuesta_ordinal(modelo)
    y_fit = y_ord[tr] if ordinal else y_ev[tr]
    balanceo, optimizador = MEJORES[modelo]
    pipe = construir_pipeline(modelo, datos)
    esp = ESPACIOS[modelo]
    scoring = PuntuadorEvento(datos["umbral"]) if ordinal else "roc_auc"
    encolar = [esp["inicial"]] if "inicial" in esp and optimizador == "Bayesiano" else None
    t0 = time.time()
    res = OPTIMIZADORES[optimizador](pipe, esp["space"], X_tr, y_fit, internas, scoring,
                                     seed=semilla + fold, presupuesto=presupuesto, encolar=encolar)
    tiempo_busqueda = time.time() - t0
    mejor = res.mejor_estimador
    t1 = time.perf_counter()
    s = puntaje(mejor, X_te)
    pred = mejor.predict(X_te)
    tiempo_pred = (time.perf_counter() - t1) / len(te)
    fila = {"conjunto": conjunto, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
            "repeticion": repeticion, "semilla": semilla, "fold": fold, "n_train": len(tr), "n_test": len(te),
            **metricas(y_ev[te], s, da_probabilidades(modelo), pred),
            "score_interno": res.mejor_score_interno, "n_evaluaciones": res.n_evaluaciones,
            "tiempo_busqueda_s": round(tiempo_busqueda, 3), "tiempo_ajuste_s": round(res.tiempo_refit, 4),
            "tiempo_prediccion_ms_por_caso": round(1000 * tiempo_pred, 5),
            "mejores_hiperparametros": repr(res.mejores_hiperparametros)}
    predicciones = pd.DataFrame({"conjunto": conjunto, "modelo": modelo, "repeticion": repeticion, "fold": fold,
                                 "idx": te, "y_evento": y_ev[te], "y_ord": y_ord[te], "puntaje": s})
    return fila, predicciones


def _anexar(df, ruta, intentos=10):
    for k in range(intentos):
        try:
            df.to_csv(ruta, mode="a", header=not Path(ruta).exists(), index=False)
            return
        except PermissionError:
            if k == intentos - 1:
                raise
            time.sleep(3)


def ejecutar_lote(tareas, ruta_filas, ruta_pred, n_procesos=11, presupuesto=PRESUPUESTO, registro=print):
    """Ejecuta en paralelo las tareas pendientes (conjunto, modelo, repetición, pliegue) y anexa cada resultado.

    Las tareas ya presentes en ``ruta_filas`` se omiten, de modo que la corrida se puede interrumpir y reanudar.
    """
    from joblib import Parallel, delayed

    ruta_filas, ruta_pred = Path(ruta_filas), Path(ruta_pred)
    hechas = set()
    if ruta_filas.exists():
        prev = pd.read_csv(ruta_filas)
        hechas = set(zip(prev["conjunto"], prev["modelo"], prev["repeticion"], prev["fold"]))
    pendientes = [t for t in tareas if tuple(t) not in hechas]
    pendientes.sort(key=lambda t: PESO_MODELO.get(t[1], 1) * (4 if t[0] in ("vino", "abalone") else 1), reverse=True)
    registro(f"{len(pendientes)} tareas pendientes de {len(tareas)}")
    if not pendientes:
        return
    t0 = time.time()
    trabajos = (delayed(ejecutar_tarea)(*t, presupuesto=presupuesto) for t in pendientes)
    for i, (fila, pred) in enumerate(Parallel(n_jobs=n_procesos, return_as="generator_unordered", batch_size=1)(trabajos), 1):
        _anexar(pd.DataFrame([fila]), ruta_filas)
        _anexar(pred, ruta_pred)
        trans = time.time() - t0
        registro(f"[{i}/{len(pendientes)}] {fila['conjunto']} | {fila['modelo']} | rep {fila['repeticion']} | "
                 f"fold {fila['fold']} | búsqueda {fila['tiempo_busqueda_s']:.0f}s | "
                 f"transcurrido {trans / 60:.1f} min | restante ~{trans / i * (len(pendientes) - i) / 60:.0f} min")
