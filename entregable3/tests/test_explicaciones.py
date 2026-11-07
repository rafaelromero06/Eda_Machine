import json

import joblib
import numpy as np
import pytest

from src.data import CATEGORICAS_VINO, NUMERICAS_VINO, RUTA_ARTEFACTOS, cargar_vinos
from src.explicaciones import Explicador

CARPETAS = sorted(p for p in RUTA_ARTEFACTOS.iterdir() if (p / "modelo.joblib").exists()) if RUTA_ARTEFACTOS.exists() else []


@pytest.mark.skipif(not CARPETAS, reason="todavía no hay artefactos serializados")
@pytest.mark.parametrize("carpeta", CARPETAS, ids=lambda p: p.name)
def test_aportes_mas_base_reproducen_la_salida(carpeta):
    modelo = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))["modelo"]
    pipe = joblib.load(carpeta / "modelo.joblib")
    vinos = cargar_vinos()[NUMERICAS_VINO + CATEGORICAS_VINO]
    exp = Explicador(modelo, pipe, vinos.sample(200, random_state=0))
    X = vinos.sample(4, random_state=1)
    aportes, base = exp.explicar_varios(X)
    assert set(aportes.columns) == set(NUMERICAS_VINO + CATEGORICAS_VINO)
    np.testing.assert_allclose(aportes.sum(axis=1).to_numpy() + base, exp.salida(X), rtol=0, atol=1e-6)
    uno, base_uno = exp.explicar(X.iloc[[2]])
    np.testing.assert_allclose(uno.to_numpy(), aportes.iloc[2][uno.index].to_numpy(), rtol=0, atol=1e-9)
    assert base_uno == pytest.approx(base)
