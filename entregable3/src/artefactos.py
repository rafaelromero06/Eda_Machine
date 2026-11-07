from __future__ import annotations

import hashlib
import json
import platform
import re
import time
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .benchmarks import ESPACIOS, MEJORES, NOMBRE_ORIGINAL, construir_pipeline, puntaje, usa_respuesta_ordinal
from .data import RUTA_BASE, cargar_conjunto, esquema_vino
from .optimizadores import OPTIMIZADORES
from .validacion import K_INTERNO, PRESUPUESTO, PuntuadorEvento, particiones

RUTA_ARTEFACTOS = RUTA_BASE / "artifacts"
SEMILLA_FINAL = 42
SEMILLA_PRUEBA_FIJA = 2026
N_PRUEBA_FIJA = 200
LIBRERIAS = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "catboost",
             "joblib", "optuna"]


def version_instalada(libreria: str) -> str:
    """Versión de una librería; acepta la variante sin GPU (por ejemplo, xgboost-cpu en lugar de xgboost)."""
    for nombre in (libreria, f"{libreria}-cpu"):
        try:
            return version(nombre)
        except PackageNotFoundError:
            continue
    return "no instalada"


def slug(nombre: str) -> str:
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def a_json(v):
    """Convierte un valor de hiperparámetro a un tipo que JSON acepta (inf se guarda como texto)."""
    if v is None or isinstance(v, (str, bool)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else str(float(v))
    return str(v)


def conjunto_prueba_fijo():
    """200 vinos elegidos con semilla fija para la prueba de ida y vuelta."""
    datos = cargar_conjunto("vino")
    idx = np.sort(np.random.default_rng(SEMILLA_PRUEBA_FIJA).choice(len(datos["y_ord"]), N_PRUEBA_FIJA, replace=False))
    return idx, datos["X"].iloc[idx]


def buscar_final(modelo: str, presupuesto=PRESUPUESTO):
    """Búsqueda de hiperparámetros sobre los 5320 vinos (3 pliegues internos) y reajuste con todos los datos."""
    datos = cargar_conjunto("vino")
    ordinal = usa_respuesta_ordinal(modelo)
    y_fit = datos["y_ord"] if ordinal else datos["y_evento"]
    balanceo, optimizador = MEJORES[modelo]
    esp = ESPACIOS[modelo]
    encolar = [esp["inicial"]] if "inicial" in esp and optimizador == "Bayesiano" else None
    res = OPTIMIZADORES[optimizador](construir_pipeline(modelo, datos), esp["space"], datos["X"], y_fit,
                                     particiones(datos["y_evento"], SEMILLA_FINAL, K_INTERNO),
                                     PuntuadorEvento(datos["umbral"]) if ordinal else "roc_auc",
                                     seed=SEMILLA_FINAL, presupuesto=presupuesto, encolar=encolar)
    return res


def reajustar(modelo: str, hiperparametros: dict, **cambios):
    """Pipeline final con hiperparámetros dados, ajustado con los 5320 vinos."""
    datos = cargar_conjunto("vino")
    pipe = construir_pipeline(modelo, datos)
    pipe.set_params(**{f"modelo__{k}": v for k, v in {**hiperparametros, **cambios}.items()})
    t0 = time.perf_counter()
    pipe.fit(datos["X"], datos["y_ord"] if usa_respuesta_ordinal(modelo) else datos["y_evento"])
    return pipe, time.perf_counter() - t0


def guardar(modelo: str, pipe, hiperparametros: dict, score_interno: float, metricas_externas: dict,
            tiempo_ajuste: float, extra: dict | None = None) -> dict:
    """Serializa el pipeline, verifica la ida y vuelta y escribe metadata.json."""
    carpeta = RUTA_ARTEFACTOS / slug(modelo)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "modelo.joblib"
    idx, X_fijo = conjunto_prueba_fijo()
    en_memoria = puntaje(pipe, X_fijo)
    joblib.dump(pipe, ruta, compress=3)
    cargado = joblib.load(ruta)
    recargado = puntaje(cargado, X_fijo)
    pd.DataFrame({"idx": idx, "puntaje": en_memoria}).to_csv(carpeta / "prediccion_referencia.csv", index=False,
                                                            float_format="%.17g")
    balanceo, optimizador = MEJORES[modelo]
    meta = {
        "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador,
        "tarea": "clasificación: vino bueno (quality >= 7)", "umbral_evento": 7,
        "entrena_con": "nota ordinal (quality)" if usa_respuesta_ordinal(modelo) else "evento binario",
        "semilla": SEMILLA_FINAL, "presupuesto_evaluaciones": PRESUPUESTO,
        "hiperparametros_finales": {k: a_json(v) for k, v in hiperparametros.items()},
        "score_interno_auc": float(score_interno),
        "metricas_bucle_externo": metricas_externas,
        "esquema_entrada": esquema_vino(),
        "salida": "probabilidad de vino bueno" if modelo != "SVM" else "función de decisión (sin probabilidad)",
        "versiones": {"python": platform.python_version(), **{lib: version_instalada(lib) for lib in LIBRERIAS}},
        "fecha": time.strftime("%Y-%m-%d"),
        "tiempo_ajuste_s": round(float(tiempo_ajuste), 3),
        "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 3),
        "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
        "prueba_ida_y_vuelta": {"n_casos": int(len(idx)), "semilla": SEMILLA_PRUEBA_FIJA,
                                "diferencia_maxima": float(np.max(np.abs(en_memoria - recargado))),
                                "identicas": bool(np.array_equal(en_memoria, recargado))},
        **(extra or {}),
    }
    (carpeta / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def cargar_artefacto(modelo: str):
    carpeta = RUTA_ARTEFACTOS / slug(modelo)
    return joblib.load(carpeta / "modelo.joblib"), json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))


def compactar_bof(est):
    """Reduce la memoria de un Bosque Ordinal Focalizado ajustado (enteros de 32 bits y valores float32)."""
    for nombre in ("izquierdo_", "derecho_", "raices_", "filas_nodo_"):
        setattr(est, nombre, getattr(est, nombre).astype(np.int32))
    for nombre in ("valor_", "valor_agregado_", "peso_nodo_", "puntaje_oob_"):
        setattr(est, nombre, getattr(est, nombre).astype(np.float32))
    return est
