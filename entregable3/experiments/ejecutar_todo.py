import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASOS = [
    ("validacion", ["experiments/correr_validacion.py", "todo"]),
    ("ablacion", ["experiments/ablacion.py"]),
    ("modelos_finales", ["experiments/modelos_finales.py"]),
    ("robustez", ["experiments/robustez.py", "paralelo"]),
    ("escalabilidad", ["experiments/robustez.py", "escalabilidad"]),
    ("memoria", ["experiments/memoria.py"]),
    ("analisis", ["experiments/analisis.py"]),
    ("tabla_maestra", ["experiments/tabla_maestra_e3.py"]),
    ("documentos", ["experiments/documentos.py"]),
    ("app", ["experiments/preparar_app.py"]),
]

if __name__ == "__main__":
    desde = sys.argv[1] if len(sys.argv) > 1 else PASOS[0][0]
    nombres = [n for n, _ in PASOS]
    for nombre, args in PASOS[nombres.index(desde):]:
        t0 = time.time()
        print(f"== {nombre}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=RAIZ)
        print(f"== {nombre}: código {r.returncode} en {(time.time() - t0) / 60:.1f} min", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
