import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.benchmarks import MODELOS, NOMBRE_ORIGINAL
from src.robustez import (ESCENARIOS_RUIDO, FRACCIONES, GRILLAS_SENSIBILIDAD, SEMILLAS_MODELO, cambio_distribucion,
                          escalabilidad, hiperparametros_finales, tarea, validacion_por_bloques)

RESULTADOS = RAIZ / "experiments" / "resultados"
ESTOCASTICOS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Decision Tree", NOMBRE_ORIGINAL]


def lista_tareas(hp):
    t = []
    for m in MODELOS:
        for f in range(5):
            t.append(("referencia", m, f, hp[m], None, None))
            for nombre, nivel in ESCENARIOS_RUIDO:
                t.append(("ruido", m, f, hp[m], nombre, nivel))
            for fr in FRACCIONES[:-1]:
                t.append(("tamano", m, f, hp[m], "fraccion", fr))
            t.append(("conforme", m, f, hp[m], None, None))
            if m in ESTOCASTICOS:
                for s in SEMILLAS_MODELO:
                    t.append(("semillas", m, f, hp[m], "semilla", s))
    for parametro, valores in GRILLAS_SENSIBILIDAD.items():
        for v in valores:
            for f in range(5):
                t.append(("sensibilidad", NOMBRE_ORIGINAL, f, hp[NOMBRE_ORIGINAL], parametro, v))
    return t


if __name__ == "__main__":
    hp = hiperparametros_finales(RESULTADOS / "hiperparametros_finales.csv")
    parte = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if parte in ("todo", "paralelo"):
        tareas = lista_tareas(hp)
        print(f"{len(tareas)} tareas de robustez", flush=True)
        t0 = time.time()
        res = Parallel(n_jobs=11, batch_size=1, verbose=5)(delayed(tarea)(*t) for t in tareas)
        filas = pd.DataFrame([f for lista in res for f in lista])
        filas.to_csv(RESULTADOS / "robustez_filas.csv", index=False)
        cambio = Parallel(n_jobs=10)(delayed(cambio_distribucion)(m, hp[m], o) for m in MODELOS for o in ("red", "white"))
        pd.DataFrame([f for lista in cambio for f in lista]).to_csv(RESULTADOS / "robustez_cambio.csv", index=False)
        bloques = Parallel(n_jobs=10)(delayed(validacion_por_bloques)(m, hp[m]) for m in MODELOS)
        pd.DataFrame([f for lista in bloques for f in lista]).to_csv(RESULTADOS / "robustez_bloques.csv", index=False)
        print(f"paralelo listo en {(time.time() - t0) / 60:.1f} min", flush=True)
    if parte in ("todo", "escalabilidad"):
        filas = [f for m in MODELOS for f in escalabilidad(m, hp[m])]
        pd.DataFrame(filas).to_csv(RESULTADOS / "robustez_escalabilidad.csv", index=False)
        print("escalabilidad lista", flush=True)
