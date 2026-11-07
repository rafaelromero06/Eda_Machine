import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import RUTA_ARTEFACTOS, cargar_conjunto

CARPETAS = sorted(p for p in RUTA_ARTEFACTOS.iterdir() if (p / "modelo.joblib").exists()) if RUTA_ARTEFACTOS.exists() else []
CLAVES = {"modelo", "versiones", "semilla", "hiperparametros_finales", "esquema_entrada", "metricas_bucle_externo",
          "prueba_ida_y_vuelta", "sha256"}


@pytest.mark.skipif(not CARPETAS, reason="todavía no hay artefactos serializados")
@pytest.mark.parametrize("carpeta", CARPETAS, ids=lambda p: p.name)
def test_ida_y_vuelta(carpeta):
    """El artefacto recargado reproduce las predicciones de referencia.

    La tolerancia admite la precisión float32 de los artefactos compactados, cuyo último dígito
    cambia entre sistemas operativos; un artefacto mal guardado difiere en órdenes de magnitud más.
    """
    pipe = joblib.load(carpeta / "modelo.joblib")
    ref = pd.read_csv(carpeta / "prediccion_referencia.csv")
    X = cargar_conjunto("vino")["X"].iloc[ref["idx"].to_numpy()]
    modelo = pipe.named_steps["modelo"]
    s = pipe.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else pipe.decision_function(X)
    np.testing.assert_allclose(s, ref["puntaje"].to_numpy(), rtol=0, atol=1e-6)


@pytest.mark.skipif(not CARPETAS, reason="todavía no hay artefactos serializados")
@pytest.mark.parametrize("carpeta", CARPETAS, ids=lambda p: p.name)
def test_metadata_completa(carpeta):
    meta = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    assert CLAVES <= set(meta)
    assert meta["prueba_ida_y_vuelta"]["identicas"]
    esquema = meta["esquema_entrada"]
    assert len(esquema) == 12
    for nombre, e in esquema.items():
        if e["tipo"] == "float":
            assert e["minimo_valido"] <= e["minimo_observado"] <= e["maximo_observado"] <= e["maximo_valido"]
