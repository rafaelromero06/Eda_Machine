import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.benchmarks import NOMBRE_ORIGINAL, construir_pipeline, puntaje
from src.data import cargar_conjunto
from src.validacion import SEMILLAS_REPETICION, metricas, particiones

RESULTADOS = RAIZ / "experiments" / "resultados"
VARIANTES = {
    "completo": {},
    "sin_agregacion": {"ancho_agregacion": 0.0},
    "agregacion_uniforme": {"ancho_agregacion": np.inf},
    "particion_binaria": {"ancho_particion": 0.0},
    "sin_ordinal": {"binario": True},
    "sin_calibracion": {"calibrar": False},
}


def correr(conjunto, repeticion, fold, hiper, variante):
    datos = cargar_conjunto(conjunto)
    semilla = SEMILLAS_REPETICION[repeticion]
    tr, te = particiones(datos["y_evento"], semilla)[fold]
    cambios = dict(VARIANTES[variante])
    binario = cambios.pop("binario", False)
    pipe = construir_pipeline(NOMBRE_ORIGINAL, datos)
    pipe.set_params(**{f"modelo__{k}": v for k, v in {**hiper, **cambios}.items()})
    if binario:
        pipe.set_params(modelo__umbral=None)
    y_fit = datos["y_evento"][tr] if binario else datos["y_ord"][tr]
    t0 = time.time()
    pipe.fit(datos["X"].iloc[tr], y_fit)
    s = puntaje(pipe, datos["X"].iloc[te])
    fila = {"conjunto": conjunto, "variante": variante, "repeticion": repeticion, "fold": fold,
            **metricas(datos["y_evento"][te], s, True, pipe.predict(datos["X"].iloc[te])),
            "tiempo_ajuste_s": round(time.time() - t0, 3)}
    pred = pd.DataFrame({"conjunto": conjunto, "variante": variante, "repeticion": repeticion, "fold": fold,
                         "idx": te, "y_evento": datos["y_evento"][te], "puntaje": s})
    return fila, pred


if __name__ == "__main__":
    filas_cv = pd.read_csv(RESULTADOS / "cv_filas.csv")
    bof = filas_cv[filas_cv["modelo"] == NOMBRE_ORIGINAL]
    tareas = [(r.conjunto, int(r.repeticion), int(r.fold), ast.literal_eval(r.mejores_hiperparametros), v)
              for r in bof.itertuples() for v in VARIANTES]
    res = Parallel(n_jobs=11, batch_size=1)(delayed(correr)(*t) for t in tareas)
    pd.DataFrame([f for f, _ in res]).to_csv(RESULTADOS / "ablacion_filas.csv", index=False)
    pd.concat([p for _, p in res]).to_csv(RESULTADOS / "ablacion_predicciones.csv.gz", index=False, compression="gzip")
    print(f"{len(res)} ajustes de ablación guardados")
