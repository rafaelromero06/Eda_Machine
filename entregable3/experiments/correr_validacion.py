import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.benchmarks import MODELOS
from src.data import ADICIONALES
from src.validacion import K_EXTERNO, SEMILLAS_REPETICION, ejecutar_lote

RESULTADOS = RAIZ / "experiments" / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)
REGISTRO = RESULTADOS / "registro_validacion.log"


def registro(texto):
    linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {texto}"
    print(linea, flush=True)
    with open(REGISTRO, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def tareas(cual):
    vino = [("vino", m, r, f) for m in MODELOS for r in range(len(SEMILLAS_REPETICION)) for f in range(K_EXTERNO)]
    adicionales = [(c, m, 0, f) for c in ADICIONALES for m in MODELOS for f in range(K_EXTERNO)]
    return {"vino": vino, "adicionales": adicionales, "todo": vino + adicionales}[cual]


if __name__ == "__main__":
    cual = sys.argv[1] if len(sys.argv) > 1 else "todo"
    n_procesos = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    registro(f"inicio: {cual} con {n_procesos} procesos")
    ejecutar_lote(tareas(cual), RESULTADOS / "cv_filas.csv", RESULTADOS / "cv_predicciones.csv",
                  n_procesos=n_procesos, registro=registro)
    registro("fin")
