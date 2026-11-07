import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.analisis import RESULTADOS, correr_todo, robustez

if __name__ == "__main__":
    veredicto = correr_todo()
    print(json.dumps(veredicto, indent=1, ensure_ascii=False))
    if (RESULTADOS / "robustez_filas.csv").exists():
        robustez()
        print("tablas de robustez listas")
