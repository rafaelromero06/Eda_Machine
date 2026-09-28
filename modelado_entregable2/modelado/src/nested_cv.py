
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score,
)

from . import config
from .model_factory import obtener_espacios
from .optimizers import OPTIMIZADORES
from .pipeline_builder import construir_pipeline


def _metricas_clasificacion(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }


def _metricas_regresion(y_true, y_pred) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _score_clasificacion(pipe, X):
    """predict_proba si está disponible; si no, decision_function (SVM)."""
    modelo = pipe.named_steps["modelo"]
    if hasattr(modelo, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def ejecutar_combinacion(tarea, nombre_modelo, nombre_balanceo, nombre_optimizador,
                          X, y, k_outer=None, k_inner=None, presupuesto=None,
                          seed=None, n_jobs=config.N_JOBS, verbose=True,
                          folds_completados=frozenset()):
    """GENERADOR: yield una fila (dict) apenas se completa cada fold
    externo (en vez de devolver la lista solo al final). Así, si el
    proceso se corta a mitad de una combinación (por el límite de tiempo
    de una sola llamada), los folds ya calculados quedan guardados y no
    hay que repetirlos -- ver `ejecutar_lote`, que anexa cada fila al CSV
    apenas se produce.

    `folds_completados`: índices de fold externo que ya están en el log
    (se saltan, no se recalculan)."""
    k_outer = k_outer or config.K_OUTER
    k_inner = k_inner or config.K_INNER
    presupuesto = presupuesto or config.PRESUPUESTO_EVALUACIONES
    seed = config.RANDOM_STATE if seed is None else seed

    pipe_template, aplica = construir_pipeline(tarea, nombre_modelo, nombre_balanceo)
    if not aplica:
        if verbose:
            print(f"  [omitido: {nombre_modelo} x {nombre_balanceo} no aplica]")
        return

    space = obtener_espacios(tarea)[nombre_modelo]["space"]
    optimizador_fn = OPTIMIZADORES[nombre_optimizador]
    scoring = "roc_auc" if tarea == "clasificacion" else "neg_root_mean_squared_error"

    cv_cls = StratifiedKFold if tarea == "clasificacion" else KFold
    outer_cv = cv_cls(n_splits=k_outer, shuffle=True, random_state=seed)
    inner_cv = cv_cls(n_splits=k_inner, shuffle=True, random_state=seed)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold_i, (idx_tr, idx_te) in enumerate(outer_cv.split(X, y)):
        if fold_i in folds_completados:
            continue
        X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
        y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]

        t0 = time.time()
        resultado = optimizador_fn(pipe_template, space, X_tr, y_tr, inner_cv, scoring,
                                    seed=seed + fold_i, presupuesto=presupuesto, n_jobs=n_jobs)
        tiempo_fold = time.time() - t0

        mejor_pipe = resultado.mejor_estimador
        y_pred = mejor_pipe.predict(X_te)

        if tarea == "clasificacion":
            y_score = _score_clasificacion(mejor_pipe, X_te)
            metricas = _metricas_clasificacion(y_te, y_pred, y_score)
            metrica_show = metricas["auc"]
        else:
            metricas = _metricas_regresion(y_te, y_pred)
            metrica_show = metricas["rmse"]

        fila = {
            "tarea": tarea, "modelo": nombre_modelo, "balanceo": nombre_balanceo,
            "optimizador": nombre_optimizador, "fold_externo": fold_i,
            **metricas,
            "score_interno": resultado.mejor_score_interno,
            "n_evaluaciones": resultado.n_evaluaciones,
            "tiempo_seg": round(tiempo_fold, 3),
            "mejores_hiperparametros": str(resultado.mejores_hiperparametros),
            "semilla": seed + fold_i,
        }
        if verbose:
            print(f"    fold {fold_i + 1}/{k_outer} | {tiempo_fold:6.1f}s | "
                  f"n_eval={resultado.n_evaluaciones:3d} | métrica_test={metrica_show:.4f}", flush=True)
        yield fila


def ejecutar_lote(tarea, combinaciones, X, y, ruta_log: Path, reanudar=True,
                   tiempo_limite_seg=None, **kwargs):
    """Corre `ejecutar_combinacion` para una lista de tuplas
    (modelo, balanceo, optimizador), anexando cada FILA (fold) al CSV
    maestro apenas se calcula (Sección 7.3: registro estructurado de
    experimentos) -- no espera a que termine la combinación completa.

    `tiempo_limite_seg`: si se da, la función se detiene (de forma
    ordenada, sin perder ninguna fila ya escrita) apenas se supera este
    tiempo total, para poder correr la grilla en bloques con límite de
    tiempo y reanudar en la siguiente llamada."""
    ruta_log = Path(ruta_log)
    k_outer = kwargs.get("k_outer") or config.K_OUTER
    t_inicio = time.time()

    log_previo = pd.read_csv(ruta_log) if (reanudar and ruta_log.exists()) else pd.DataFrame()

    for nombre_modelo, nombre_balanceo, nombre_optimizador in combinaciones:
        if tiempo_limite_seg and (time.time() - t_inicio) > tiempo_limite_seg:
            print(f"[límite de tiempo del bloque alcanzado, {time.time()-t_inicio:.0f}s -- se corta aquí]")
            break

        folds_hechos = set()
        if not log_previo.empty:
            ya = log_previo[
                (log_previo["modelo"] == nombre_modelo) &
                (log_previo["balanceo"] == nombre_balanceo) &
                (log_previo["optimizador"] == nombre_optimizador)
            ]
            folds_hechos = set(int(f) for f in ya["fold_externo"].tolist())
        if len(folds_hechos) >= k_outer:
            continue  # ya completo, no imprime nada (evita saturar el log en corridas largas)

        faltan = sorted(set(range(k_outer)) - folds_hechos)
        print(f"[corriendo] {nombre_modelo} x {nombre_balanceo} x {nombre_optimizador} "
              f"(faltan folds {faltan})", flush=True)

        for fila in ejecutar_combinacion(tarea, nombre_modelo, nombre_balanceo, nombre_optimizador,
                                          X, y, folds_completados=folds_hechos, **kwargs):
            df_fila = pd.DataFrame([fila])
            escribir_header = not ruta_log.exists()
            df_fila.to_csv(ruta_log, mode="a", header=escribir_header, index=False)
            log_previo = pd.concat([log_previo, df_fila], ignore_index=True) if not log_previo.empty else df_fila

            if tiempo_limite_seg and (time.time() - t_inicio) > tiempo_limite_seg:
                print(f"[límite de tiempo del bloque alcanzado a mitad de combinación, "
                      f"{time.time()-t_inicio:.0f}s -- se corta aquí, este fold ya quedó guardado]")
                return log_previo

    return log_previo if not log_previo.empty else pd.DataFrame()


# =============================================================================
# Ejecución en PARALELO (PC con varios núcleos)
# =============================================================================
# Cada tarea = un fold externo de una combinación (modelo, balanceo,
# optimizador). Las tareas son independientes entre sí (cada una recibe su
# propio split externo, determinado por la semilla), así que se reparten
# entre N_WORKERS procesos. El proceso principal es el ÚNICO que escribe en
# los CSV (a medida que cada tarea termina), así no hay dos procesos
# escribiendo el mismo archivo a la vez y el checkpoint sigue siendo por
# fold: si la corrida se interrumpe, volver a lanzarla retoma lo pendiente.

# Peso relativo aproximado de cada tarea, medido en la corrida de 1 núcleo.
# Solo se usa para ORDENAR: las tareas más pesadas se lanzan primero, así
# no queda un Random Forest largo corriendo solo al final mientras el resto
# de núcleos ya terminó (reduce el tiempo total de la corrida).
_PESO_MODELO = {"Random Forest": 10, "XGBoost": 5, "SVM": 3, "SVR": 3, "KNN": 1.5}
_PESO_BALANCEO = {"SMOTE": 2, "ADASYN": 2}
_PESO_OPTIMIZADOR = {"Grid Search": 1.5}


def _peso_tarea(t) -> float:
    return (_PESO_MODELO.get(t["modelo"], 1)
            * _PESO_BALANCEO.get(t["balanceo"], 1)
            * _PESO_OPTIMIZADOR.get(t["optimizador"], 1))


def _ejecutar_fold(tarea, modelo, balanceo, optimizador, fold_i, X, y,
                   k_outer, k_inner, presupuesto, seed):
    """Corre UN fold externo. Se ejecuta dentro de un proceso worker."""
    import warnings
    warnings.filterwarnings("ignore")  # los workers no heredan los filtros del proceso principal
    otros = frozenset(range(k_outer)) - {fold_i}
    filas = list(ejecutar_combinacion(
        tarea, modelo, balanceo, optimizador, X, y,
        k_outer=k_outer, k_inner=k_inner, presupuesto=presupuesto, seed=seed,
        n_jobs=1, verbose=False, folds_completados=otros,
    ))
    return filas[0] if filas else None


def _anexar_fila(fila: dict, ruta_log: Path, intentos: int = 10):
    """Anexa una fila al CSV. Reintenta si el archivo está bloqueado un
    instante (pasa en Windows cuando OneDrive o Excel tienen el CSV abierto
    mientras se sincroniza): mejor esperar unos segundos que perder el fold."""
    for k in range(intentos):
        try:
            pd.DataFrame([fila]).to_csv(ruta_log, mode="a", header=not ruta_log.exists(), index=False)
            return
        except PermissionError:
            if k == intentos - 1:
                raise
            print(f"  [CSV bloqueado ({ruta_log.name}), reintentando en 3s... "
                  f"¿está abierto en Excel?]", flush=True)
            time.sleep(3)


def _formatear_duracion(seg: float) -> str:
    seg = int(seg)
    h, resto = divmod(seg, 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def ejecutar_en_paralelo(grupos, n_workers=None, k_outer=None, k_inner=None,
                         presupuesto=None, seed=None):
    """`grupos`: lista de dicts {"tarea", "combinaciones", "X", "y", "ruta_log"}
    (uno para clasificación y otro para regresión). Todas las tareas de
    todos los grupos van a un mismo pool, para que ningún núcleo quede
    ocioso mientras el otro grupo aún tiene trabajo."""
    from joblib import Parallel, delayed

    n_workers = n_workers or config.N_WORKERS
    k_outer = k_outer or config.K_OUTER
    k_inner = k_inner or config.K_INNER
    presupuesto = presupuesto or config.PRESUPUESTO_EVALUACIONES
    seed = config.RANDOM_STATE if seed is None else seed

    pendientes, rutas, datos = [], {}, {}
    for g in grupos:
        tarea, ruta_log = g["tarea"], Path(g["ruta_log"])
        rutas[tarea] = ruta_log
        datos[tarea] = (g["X"].reset_index(drop=True), g["y"].reset_index(drop=True))

        hechos = set()
        if ruta_log.exists():
            previo = pd.read_csv(ruta_log)
            hechos = set(zip(previo["modelo"], previo["balanceo"],
                             previo["optimizador"], previo["fold_externo"].astype(int)))

        n_no_aplica = 0
        for modelo, balanceo, optimizador in g["combinaciones"]:
            _, aplica = construir_pipeline(tarea, modelo, balanceo)
            if not aplica:
                n_no_aplica += 1
                continue
            for fold_i in range(k_outer):
                if (modelo, balanceo, optimizador, fold_i) not in hechos:
                    pendientes.append({"tarea": tarea, "modelo": modelo, "balanceo": balanceo,
                                       "optimizador": optimizador, "fold": fold_i})
        print(f"  {tarea}: {len(g['combinaciones']) - n_no_aplica} combinaciones válidas "
              f"({n_no_aplica} no aplican), {len(hechos)} folds ya hechos", flush=True)

    total = len(pendientes)
    if total == 0:
        print("\nNo hay tareas pendientes: todo está completo en los CSV.", flush=True)
        return

    pendientes.sort(key=_peso_tarea, reverse=True)
    print(f"\n{total} tareas pendientes (1 tarea = 1 fold externo) | "
          f"{n_workers} procesos en paralelo\n", flush=True)

    t0 = time.time()
    trabajos = (
        delayed(_ejecutar_fold)(t["tarea"], t["modelo"], t["balanceo"], t["optimizador"],
                                t["fold"], *datos[t["tarea"]], k_outer, k_inner, presupuesto, seed)
        for t in pendientes
    )
    resultados = Parallel(n_jobs=n_workers, return_as="generator_unordered", batch_size=1)(trabajos)

    for i, fila in enumerate(resultados, start=1):
        if fila is None:
            continue
        _anexar_fila(fila, rutas[fila["tarea"]])

        transcurrido = time.time() - t0
        restante = transcurrido / i * (total - i)
        metrica = (f"AUC={fila['auc']:.4f}" if fila["tarea"] == "clasificacion"
                   else f"RMSE={fila['rmse']:.4f}")
        print(f"[{i:3d}/{total}] {fila['modelo']} x {fila['balanceo']} x {fila['optimizador']} "
              f"| fold {fila['fold_externo'] + 1} | {metrica} | {fila['tiempo_seg']:.0f}s "
              f"| transcurrido {_formatear_duracion(transcurrido)}, "
              f"restante ~{_formatear_duracion(restante)}", flush=True)

    print(f"\nListo: {total} tareas en {_formatear_duracion(time.time() - t0)}.", flush=True)
