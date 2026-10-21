from __future__ import annotations

import os
import platform
import sys
from importlib import metadata

import pandas as pd

from . import config

PAQUETES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "optuna", "deap",
            "joblib", "statsmodels", "scikit-posthocs", "shap", "lime", "matplotlib", "seaborn"]
RUTA_ENTORNO = config.RUTA_EXPERIMENTOS / "entorno.txt"


def version(paquete: str) -> str:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return "no instalado"


def versiones(paquetes=None) -> pd.DataFrame:
    """Versión de Python y de cada paquete, en una tabla de dos columnas."""
    filas = [("python", platform.python_version())]
    filas += [(p, version(p)) for p in (paquetes or PAQUETES)]
    return pd.DataFrame(filas, columns=["paquete", "versión"])


def nombre_procesador() -> str:
    """Nombre comercial del procesador (en Windows lo lee del registro)."""
    if sys.platform == "win32":
        try:
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def equipo() -> dict:
    """Sistema operativo, procesador, núcleos y memoria del equipo."""
    info = {"sistema": platform.platform(), "procesador": nombre_procesador(), "núcleos lógicos": os.cpu_count()}
    try:
        import psutil
        info["núcleos físicos"] = psutil.cpu_count(logical=False)
        info["memoria RAM (GB)"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    info["procesos en paralelo (N_WORKERS)"] = config.N_WORKERS
    return info


def guardar_entorno(ruta=RUTA_ENTORNO) -> None:
    """Escribe Python, equipo y la lista completa de paquetes instalados (``nombre==versión``)."""
    paquetes = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                       if d.metadata["Name"]}, key=str.lower)
    lineas = [f"# Python {platform.python_version()}", *[f"# {k}: {v}" for k, v in equipo().items()], *paquetes]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
