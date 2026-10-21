import sys
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src import config
from src.data_prep import preparar_dataset, separar_X_y
from src.predicciones import (RUTA_PREDICCIONES, cargar_oof, generar_oof, metricas_por_fold, ruta_oof)

METRICAS = {"clasificacion": ["auc", "accuracy", "precision", "recall", "f1"], "regresion": ["rmse", "mae"]}


def verificar(tarea: str, oof: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Compara las métricas recalculadas desde el OOF con las del log."""
    claves = ["modelo", "balanceo", "optimizador", "fold_externo"]
    recalculadas = metricas_por_fold(oof, tarea)
    juntas = log[claves + METRICAS[tarea]].merge(recalculadas, on=claves, suffixes=("_log", "_oof"))
    filas = [{"tarea": tarea, "metrica": m, "filas": len(juntas),
              "max_diferencia": float((juntas[f"{m}_log"] - juntas[f"{m}_oof"]).abs().max())}
             for m in METRICAS[tarea]]
    return pd.DataFrame(filas)


def main():
    forzar = "--forzar" in sys.argv
    RUTA_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    X, y_reg, y_clf = separar_X_y(preparar_dataset(verbose=False))
    objetivos = {"clasificacion": y_clf, "regresion": y_reg}
    logs = {"clasificacion": config.RUTA_LOG_CLASIFICACION, "regresion": config.RUTA_LOG_REGRESION}

    tiempos, verificaciones = [], []
    for tarea in ["clasificacion", "regresion"]:
        destino = ruta_oof(tarea)
        log = pd.read_csv(logs[tarea])
        if destino.exists() and not forzar:
            print(f"{tarea}: ya existe {destino.name}, se omite (usar --forzar para recalcular)")
            oof = cargar_oof(tarea)
        else:
            print(f"{tarea}: reentrenando {len(log)} tareas con {config.N_WORKERS} procesos…", flush=True)
            t0 = time.time()
            oof, t = generar_oof(tarea, log, X, objetivos[tarea])
            t["tarea"] = tarea
            tiempos.append(t)
            oof.to_csv(destino, index=False, compression="gzip")
            print(f"{tarea}: listo en {time.time() - t0:.0f} s -> {destino.name}", flush=True)
        verificaciones.append(verificar(tarea, oof, log))

    if tiempos:
        pd.concat(tiempos).to_csv(RUTA_PREDICCIONES / "tiempos_reentrenamiento.csv", index=False)
    verificacion = pd.concat(verificaciones, ignore_index=True)
    verificacion.to_csv(RUTA_PREDICCIONES / "verificacion_contra_log.csv", index=False)
    print("\nDiferencia máxima entre las métricas del log y las recalculadas:")
    print(verificacion.to_string(index=False))


if __name__ == "__main__":
    main()
