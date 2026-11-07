import numpy as np
import pandas as pd
import pytest

from src.data import NUMERICAS_VINO, cargar_vinos, esquema_vino
from src.prediccion import COLUMNAS, armar_vino, disponibles, evaluar, evaluar_lote, gemelos, palancas, validar
from src.tablero import numero, unir

ESQUEMA = esquema_vino()
MEDIANAS = [ESQUEMA[c]["mediana"] for c in NUMERICAS_VINO]


def test_validar_acepta_un_vino_tipico():
    assert validar(MEDIANAS) == []


def test_validar_explica_cada_error():
    valores = list(MEDIANAS)
    valores[0], valores[1], valores[-1] = None, "abc", 99
    errores = validar(valores)
    assert len(errores) == 3
    assert "falta el valor" in errores[0]
    assert "no es un número" in errores[1]
    assert "fuera del rango válido" in errores[2]


def test_gemelos_no_incluye_al_mismo_vino():
    vino = cargar_vinos().iloc[0]
    g = gemelos([vino[c] for c in NUMERICAS_VINO], vino["tipo_vino"], k=5)
    assert len(g["notas"]) == 5 and 0 <= g["buenos"] <= 5 and g["distancia_media"] > 0


def test_numero_y_unir():
    assert numero(5320) == "5320"
    assert numero(12345.678, 1) == "12 345.7"
    assert numero(-0.5, 2) == "−0.50"
    assert unir([]) == "" and unir(["a"]) == "a" and unir(["a", "b", "c"]) == "a, b y c"


@pytest.mark.skipif(not disponibles(), reason="todavía no hay artefactos serializados")
def test_evaluar_entrega_probabilidad_y_conjunto():
    ev = evaluar(disponibles()[0], armar_vino(MEDIANAS, "white"))
    assert 0 <= ev["p"] <= 1 and ev["clase"] in ("bueno", "no bueno")
    assert set(ev["conjunto"]) <= {"bueno", "no bueno"}


@pytest.mark.skipif(not disponibles(), reason="todavía no hay artefactos serializados")
def test_palancas_suben_la_probabilidad():
    for p in palancas(disponibles()[0], armar_vino(MEDIANAS, "white")):
        assert p["p_hasta"] > p["p_desde"] and p["variable"] in NUMERICAS_VINO


@pytest.mark.skipif(not disponibles(), reason="todavía no hay artefactos serializados")
def test_lote_ordena_y_reporta_filas_invalidas():
    lote = cargar_vinos().sample(6, random_state=3)[COLUMNAS].reset_index(drop=True)
    lote["tipo_vino"] = lote["tipo_vino"].map({"red": "tinto", "white": "Blanco"})
    lote.loc[2, "alcohol"] = 40
    resultado, errores = evaluar_lote(disponibles()[0], lote)
    assert len(resultado) == 5 and len(errores) == 1
    assert np.all(np.diff(resultado["probabilidad_bueno"].to_numpy()) <= 0)
    assert 3 not in set(resultado["fila"])


def test_lote_sin_columnas_se_rechaza():
    resultado, errores = evaluar_lote("Bosque Ordinal Focalizado", pd.DataFrame({"alcohol": [10.0]}))
    assert resultado is None and "Faltan columnas" in errores[0]
