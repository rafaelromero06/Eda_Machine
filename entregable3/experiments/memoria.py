import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.benchmarks import MODELOS

RESULTADOS = RAIZ / "experiments" / "resultados"

CODIGO = r"""
import sys, json, pickle, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {raiz!r})
import psutil
from src.robustez import hiperparametros_finales, ajustar
from src.data import cargar_conjunto
proceso = psutil.Process()
datos = cargar_conjunto("vino")
hp = hiperparametros_finales({ruta!r})[{modelo!r}]
base = proceso.memory_info().rss
t0 = time.perf_counter()
pipe, seg = ajustar({modelo!r}, hp, datos["X"], datos["y_ord"], datos["y_evento"], datos)
pico = getattr(proceso.memory_info(), "peak_wset", proceso.memory_info().rss)
print(json.dumps({{"modelo": {modelo!r}, "memoria_base_mb": base / 2**20, "memoria_pico_mb": pico / 2**20,
                  "memoria_ajuste_mb": (pico - base) / 2**20, "tamano_en_memoria_mb": len(pickle.dumps(pipe)) / 2**20,
                  "ajuste_s": seg}}))
"""


if __name__ == "__main__":
    filas = []
    for m in MODELOS:
        codigo = CODIGO.format(raiz=str(RAIZ), ruta=str(RESULTADOS / "hiperparametros_finales.csv"), modelo=m)
        salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=3600)
        linea = [l for l in salida.stdout.splitlines() if l.startswith("{")]
        if linea:
            filas.append(json.loads(linea[-1]))
            print(filas[-1], flush=True)
        else:
            print("error", m, salida.stderr[-500:], flush=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / "memoria_modelos.csv", index=False)
