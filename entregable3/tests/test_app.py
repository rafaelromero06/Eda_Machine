import importlib

import dash
import pytest

from src.benchmarks import MODELOS


@pytest.fixture(scope="module")
def aplicacion():
    return importlib.import_module("app").app


def test_paginas_registradas(aplicacion):
    rutas = {p["path"] for p in dash.page_registry.values()}
    assert {"/", "/eda", "/modelos", "/cata", "/ficha"} <= rutas


def test_layouts_se_construyen(aplicacion):
    for pagina in dash.page_registry.values():
        layout = pagina["layout"]
        assert (layout() if callable(layout) else layout) is not None


def test_filtros_del_eda(aplicacion):
    from pages.eda import actualizar_dispersion, actualizar_general, actualizar_variable, filtrar
    assert len(filtrar(["red"], [3, 9], [8, 15])) > 0
    salida = actualizar_general(["red", "white"], [3, 9], [8.0, 15.0])
    assert len(salida) == 9
    assert "5320" in str(salida[0].to_plotly_json())
    assert "19.0 %" in str(salida[2])
    assert "son buenos" in salida[2][0].children
    figs = actualizar_variable(["white"], [3, 9], [8.0, 15.0], "alcohol")
    assert len(figs) == 3
    disp = actualizar_dispersion(["red", "white"], [3, 9], [8.0, 15.0], "alcohol", "volatile acidity", "clase")
    assert len(disp[2].frames) == 7


def test_filtros_extremos_no_rompen_el_eda(aplicacion):
    from pages.eda import actualizar_general, actualizar_variable
    assert len(actualizar_general(["red"], [9, 9], [8.0, 8.5])) == 9
    assert "por debajo y por encima de 7" in actualizar_variable(["red", "white"], [7, 9], [8.0, 15.0], "alcohol")[2]


@pytest.mark.parametrize("pestana", ["lideres", "prediccion", "diagnostico", "comparacion", "original"])
def test_pestanas_de_modelos(aplicacion, pestana):
    from pages.modelos import renderizar
    assert renderizar(pestana) is not None


def test_pestana_desde_la_url(aplicacion):
    from pages.modelos import pestana_desde_url
    assert pestana_desde_url("?pestana=diagnostico") == "diagnostico"
    assert pestana_desde_url("?pestana=inventada") == "lideres"
    assert pestana_desde_url(None) == "lideres"


@pytest.mark.parametrize("tarea", ["clasificacion", "regresion"])
def test_torneo_del_entregable_2(aplicacion, tarea):
    from pages.modelos import actualizar_torneo
    from src.data import tabla_app
    if tabla_app("e2_clasificacion.csv") is None:
        pytest.skip("todavía no hay datos del Entregable 2 en data/app")
    fig, texto = actualizar_torneo(tarea)
    assert fig.data and texto


@pytest.mark.parametrize("modelo", MODELOS)
def test_diagnostico_de_cada_modelo(aplicacion, modelo):
    from pages.modelos import diagnostico
    from src.data import tabla_app
    oof = tabla_app("oof_rep1.csv.gz")
    if oof is None or modelo not in oof.columns:
        pytest.skip("todavía no hay predicciones fuera de muestra de este modelo")
    assert len(diagnostico(modelo)) == 7


@pytest.mark.parametrize("modelo", MODELOS)
def test_prediccion_con_un_vino_valido(aplicacion, modelo):
    from pages.modelos import explicar_prediccion, predecir
    from src.data import NUMERICAS_VINO, esquema_vino
    from src.prediccion import disponibles
    if modelo not in disponibles():
        pytest.skip("todavía no hay artefacto de este modelo")
    e = esquema_vino()
    datos, resumen, alerta = predecir([e[c]["mediana"] for c in NUMERICAS_VINO], "white", modelo)
    assert alerta is None and 0 <= datos["p"] <= 1
    hijos = explicar_prediccion(datos)
    assert len(hijos) >= 2
    assert not any(type(h).__name__ == "Alert" for h in hijos)


def test_prediccion_avisa_entradas_invalidas(aplicacion):
    from pages.modelos import predecir
    from src.data import NUMERICAS_VINO
    from src.prediccion import disponibles
    if not disponibles():
        pytest.skip("todavía no hay artefactos")
    datos, resumen, alerta = predecir([None] * len(NUMERICAS_VINO), "white", disponibles()[0])
    assert datos is dash.no_update and type(alerta).__name__ == "Alert"


def test_comparacion_con_cada_rival(aplicacion):
    from pages.modelos import comparar
    from src.data import tabla_app
    pr = tabla_app("pruebas_bof.csv")
    if pr is None:
        pytest.skip("todavía no hay pruebas de hipótesis")
    for rival in pr["benchmark"]:
        duelo, jueces = comparar(rival)
        assert duelo and jueces


def test_glosario_agrupado_y_completo(aplicacion):
    from src.tablero import GLOSARIO, GRUPOS_GLOSARIO, panel_glosario
    agrupadas = [c for claves in GRUPOS_GLOSARIO.values() for c in claves]
    assert sorted(agrupadas) == sorted(GLOSARIO)
    assert len(agrupadas) == len(set(agrupadas))
    assert panel_glosario()


def test_recorrido_marca_la_pagina_actual(aplicacion):
    from src.tablero import PAGINAS, recorrido, siguiente
    rutas = {p["path"] for p in dash.page_registry.values()}
    assert {p[0] for p in PAGINAS} == rutas
    for ruta, *_ in PAGINAS:
        assert str(recorrido(ruta).to_plotly_json()).count("eslabon actual") == 1
        assert siguiente(ruta) is not None


def test_nombres_en_frases_respetan_simbolos():
    from src.tablero import minuscula, p_valor
    assert minuscula("Alcohol") == "alcohol"
    assert minuscula("SO₂ libre") == "SO₂ libre"
    assert minuscula("pH") == "pH"
    assert p_valor(2.6e-101) == "< 0.0001"
    assert p_valor(0.0451) == "0.0451"
