from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import numpy as np
from dash import dash_table, dcc, html

from . import plots
from .data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos, json_app, tabla_app
from .prediccion import ESQUEMA, ORIGINAL, disponibles
from .tablero import (CORTOS, NOMBRES, como_leer, encabezado, fila, idea, lectura, minuscula, numero, p_valor, pasos, pendiente, tabla, tarjeta,
                      termino)

BALANCEOS = {"sin_balanceo": "Sin balanceo", "class_weight": "Pesos por clase", "SMOTE": "SMOTE", "ADASYN": "ADASYN",
             "no aplica (usa la nota ordinal)": "No aplica (usa la nota completa)", "ponderación en la búsqueda": "Pesos por clase (se decide en la búsqueda)"}
OPTIMIZADORES = {"Grid Search": "Grid", "Random Search": "Random", "Bayesiano": "Bayes", "Genético": "Genético"}
FORMATO_3 = dash_table.Format.Format(precision=3, scheme=dash_table.Format.Scheme.fixed)
FORMATO_4 = dash_table.Format.Format(precision=4, scheme=dash_table.Format.Scheme.fixed)
ESTILO_TABLA = dict(
    style_table={"overflowX": "auto"},
    style_cell={"fontFamily": "'Instrument Sans', system-ui, sans-serif", "fontSize": 14, "padding": "8px 10px", "textAlign": "left",
                "fontVariantNumeric": "tabular-nums", "backgroundColor": "#fffdf8", "border": "none", "borderBottom": "1px solid #e6decd",
                "whiteSpace": "normal", "height": "auto"},
    style_header={"fontWeight": 650, "backgroundColor": "#f6f1e6", "color": "#4a5570", "borderBottom": "2px solid #e6decd"},
    style_filter={"backgroundColor": "#ffffff"},
    tooltip_delay=0, tooltip_duration=None,
)
LEGIBLE = {"BOF vs CB": "modelo original frente a CatBoost", "a favor del BOF": "a favor del modelo original", "H0": "la hipótesis nula"}


def legible(texto):
    """Cambia las siglas internas de las tablas por nombres que entiende cualquier lector."""
    for sigla, largo in LEGIBLE.items():
        texto = str(texto).replace(sigla, largo)
    return texto


def ir_a(pestana, texto):
    """Botón que lleva a la pestaña siguiente del recorrido de modelos."""
    return html.Div(html.Button([texto, html.I(className="bi bi-arrow-right")], id={"tipo": "ir", "pestana": pestana}, type="button", className="boton"),
                    className="ir-siguiente")


def _prevalencia():
    d = json_app("decision.json")
    return d["positivos"] / (d["positivos"] + d["negativos"]) if d else 0.19


def _nan_a_nulo(valor):
    return None if valor is None or (isinstance(valor, float) and not np.isfinite(valor)) else valor


def tab_lideres():
    t = tabla_app("lideres.csv")
    if t is None:
        return pendiente("La tabla de líderes")
    filas = [{"modelo": r.modelo, "corto": CORTOS.get(r.modelo, r.modelo), "nombre": NOMBRES.get(r.modelo, r.modelo),
              "auc_media": _nan_a_nulo(r.auc_media), "pr_auc_media": _nan_a_nulo(r.pr_auc_media), "brier_media": _nan_a_nulo(r.brier_media),
              "destacado": r.modelo == ORIGINAL} for r in t.itertuples()]
    bof = t[t["modelo"] == ORIGINAL].iloc[0]
    orden = t.sort_values("auc_media", ascending=False)
    mejor = orden[orden["modelo"] != ORIGINAL].iloc[0]
    puesto = int(orden["modelo"].tolist().index(ORIGINAL)) + 1
    empatados = int((orden["auc_media"].iloc[0] - orden["auc_media"] < orden["auc_sd"].iloc[0]).sum())
    datos_tabla = t.assign(ic=t.apply(lambda f: f"{f.auc_ic_inf:.3f} a {f.auc_ic_sup:.3f}", axis=1),
                           balanceo=t["balanceo"].map(BALANCEOS).fillna(t["balanceo"]))
    columnas = [("modelo", "Modelo", "text", None, "Nombre del modelo"),
                ("balanceo", "Balanceo", "text", None, "Cómo se compensó que los vinos buenos son minoría"),
                ("optimizador", "Optimizador", "text", None, "Método con que se buscó su mejor configuración"),
                ("auc_media", "AUC (media)", "numeric", FORMATO_4, "Promedio en los 25 exámenes. 0.5 es azar y 1 es perfecto"),
                ("auc_sd", "± DE", "numeric", FORMATO_4, "Desviación estándar: cuánto cambia el AUC de un examen a otro"),
                ("ic", "IC 95 %", "text", None, "Rango donde razonablemente está su AUC verdadero (bootstrap)"),
                ("pr_auc_media", "PR-AUC", "numeric", FORMATO_3, "Qué tan limpia es su lista de vinos buenos. Sin modelo valdría 0.19"),
                ("brier_media", "Brier", "numeric", FORMATO_3, "Error de sus probabilidades. Menos es mejor"),
                ("f1_media", "F1", "numeric", FORMATO_3, "Equilibrio entre precisión y sensibilidad con el umbral 0.5"),
                ("prediccion_ms_por_caso", "Tiempo por vino (ms)", "numeric", FORMATO_3, "Milisegundos que tarda en evaluar un vino"),
                ("tamano_mb", "Tamaño (MB)", "numeric", FORMATO_3, "Peso del modelo guardado en disco")]
    tabla_lideres = dash_table.DataTable(
        id="tabla-lideres", data=datos_tabla.to_dict("records"), sort_action="native", filter_action="native", page_action="none",
        filter_options={"placeholder_text": "Filtrar…", "case": "insensitive"},
        columns=[{"id": c, "name": n, "type": ty, **({"format": fo} if fo else {})} for c, n, ty, fo, _ in columnas],
        tooltip_header={c: ayuda for c, _, _, _, ayuda in columnas},
        style_data_conditional=[{"if": {"filter_query": f'{{modelo}} = "{ORIGINAL}"'}, "backgroundColor": "#e4edf9", "fontWeight": 650}],
        sort_by=[{"column_id": "auc_media", "direction": "desc"}], **ESTILO_TABLA)
    lect = idea(f"Los {empatados} primeros están prácticamente empatados." if empatados > 1 else "El primero se separa del resto.",
                f"El modelo original (fila azul) queda en el puesto {puesto} de 10 con un AUC de {bof.auc_media:.4f} ± {bof.auc_sd:.4f}; el mejor "
                f"rival es {mejor.modelo} con {mejor.auc_media:.4f} ± {mejor.auc_sd:.4f}. La diferencia entre ellos es menor que lo que cambia el "
                "AUC de un examen a otro (± DE): por eso la pestaña 4 usa pruebas estadísticas. Pulsa un encabezado para ordenar, pasa el cursor "
                "para ver qué mide, o filtra escribiendo, por ejemplo, ", html.Code("> 0.85"), " bajo «AUC (media)». SVM no tiene Brier porque "
                "no entrega probabilidades.")
    lentos = t.sort_values("prediccion_ms_por_caso")
    rapido, lento = lentos.iloc[0], lentos.iloc[-1]
    lect_costo = idea("Todos responden en milésimas de segundo: el tiempo no es un obstáculo.",
                      f"El más rápido es {rapido.modelo} ({rapido.prediccion_ms_por_caso:.3f} ms por vino) y el más lento {lento.modelo} "
                      f"({lento.prediccion_ms_por_caso:.2f} ms). Aun el más lento evalúa unos {numero(1000 / lento.prediccion_ms_por_caso)} "
                      "vinos por segundo.")
    return html.Div([
        encabezado("¿Qué modelo distingue mejor los vinos buenos?",
                   "Cada botella es un modelo, ya con su mejor configuración. Se llena según su puntaje: cuanto más llena, mejor; vacía "
                   "sería adivinar. Cambia lo que se mide y mira cómo se reordena la cava.", etiqueta="1 · Tabla de líderes"),
        tarjeta(None, [
            dcc.Store(id="cava-datos", data={"filas": filas, "prevalencia": _prevalencia()}),
            html.Div("Medir por…", className="campo-etiqueta"),
            dbc.RadioItems(id="cava-metrica", options=[{"label": "Ordenar bien los vinos (AUC)", "value": "auc"},
                                                       {"label": "Lista limpia de buenos (PR-AUC)", "value": "pr"},
                                                       {"label": "Probabilidades fiables (Brier)", "value": "brier"}],
                           value="auc", inline=True, class_name="cava-selector", input_class_name="chip-input", label_class_name="chip",
                           label_checked_class_name="chip-activo"),
            html.P(id="cava-explica", className="cava-explica", **{"aria-live": "polite"}),
            html.Div(id="cava", className="cava", role="list", **{"aria-label": "Modelos ordenados por la métrica elegida"}),
            html.Div([html.Span([html.I(style={"background": "#1d5fc4"}), "Bosque Ordinal Focalizado: el modelo original de este proyecto"]),
                      html.Span([html.I(style={"background": "#9aa3b2"}), "Los 9 modelos de referencia"]),
                      html.Span("El número de cada etiqueta es el puesto.")], className="leyenda cava-pie"),
        ], clase="mb-4"),
        tarjeta("La tabla completa", [tabla_lideres, lectura(lect)], clase="mb-4"),
        fila(tarjeta(None, [plots.figura(plots.tiempo_prediccion(t, ORIGINAL, NOMBRES)), lectura(lect_costo)]),
             tarjeta(None, torneo()), anchos=[5, 7]),
        ir_a("prediccion", "Siguiente: predecir un vino"),
    ])


def torneo():
    clas = tabla_app("e2_clasificacion.csv")
    if clas is None:
        return pendiente("El torneo del Entregable 2")
    return [
        html.H3("¿Cómo se eligió a estos finalistas?", className="tarjeta-titulo"),
        html.P(["Antes compitieron 7 modelos, cada uno con 4 formas de ", termino("balanceo"), " y 4 de ", termino("optimizador"),
                ". Cada baldosa es una combinación: más oscura, mejor. La ★ marca la mejor de cada modelo, que es la que llegó a la cava. "
                "Las baldosas en blanco son combinaciones que no existen (KNN no admite pesos por clase)."], className="small text-secondary mb-2"),
        dbc.RadioItems(id="torneo-tarea", options=[{"label": "Clasificación (AUC)", "value": "clasificacion"},
                                                   {"label": "Regresión de la nota (RMSE)", "value": "regresion"}],
                       value="clasificacion", inline=True, class_name="cava-selector", input_class_name="chip-input", label_class_name="chip",
                       label_checked_class_name="chip-activo"),
        plots.grafico("g-torneo", 470),
        lectura("", id="l-torneo"),
    ]


def figura_torneo(tarea):
    if tarea == "regresion":
        reg = tabla_app("e2_regresion.csv")
        reg = reg.assign(optimizador=reg["optimizador"].map(OPTIMIZADORES).fillna(reg["optimizador"]))
        fig = plots.mural(reg, "modelo", "optimizador", "rmse", "Torneo de regresión: error al predecir la nota", "RMSE", invertir=True)
        mejor = reg.sort_values("rmse").iloc[0]
        rango_modelos = reg.groupby("modelo")["rmse"].min()
        texto = idea(f"Al predecir la nota exacta, el mejor fue {mejor.modelo}.",
                     f"Se equivoca en promedio {mejor.rmse:.3f} puntos de nota (RMSE). La distancia entre el mejor y el peor modelo "
                     f"({rango_modelos.max() - rango_modelos.min():.3f}) es mucho mayor que la que separa a los optimizadores de un mismo modelo.")
        return fig, texto
    clas = tabla_app("e2_clasificacion.csv")
    clas = clas.assign(balanceo=clas["balanceo"].map(BALANCEOS).fillna(clas["balanceo"]),
                       optimizador=clas["optimizador"].map(OPTIMIZADORES).fillna(clas["optimizador"]))
    fig = plots.mural(clas, "modelo", ["balanceo", "optimizador"], "auc", "Torneo de clasificación: AUC de cada combinación", "AUC")
    entre = clas.groupby("modelo")["auc"].mean()
    dentro = clas.groupby("modelo")["auc"].agg(lambda s: s.max() - s.min()).mean()
    mejor = clas.sort_values("auc").iloc[-1]
    texto = idea("Importa mucho más qué modelo se usa que cómo se ajusta.",
                 f"Las filas cambian de color y las columnas casi nada: entre el mejor y el peor modelo hay {entre.max() - entre.min():.3f} de AUC, "
                 f"pero cambiar el balanceo o el optimizador de un mismo modelo mueve en promedio {dentro:.3f}. La mejor combinación fue "
                 f"{mejor.modelo} con {mejor.balanceo.lower()} y {mejor.optimizador} (AUC {mejor.auc:.4f}).")
    return fig, texto


def fila_variable(c, vinos):
    e = ESQUEMA[c]
    bajo, alto = (float(v) for v in vinos[c].quantile([0.005, 0.995]))
    paso = float(10 ** np.floor(np.log10((alto - bajo) / 200)))
    id_numero = {"tipo": "num", "var": c}
    return html.Div([
        html.Label([e["etiqueta"], html.Small(e["unidad"])], htmlFor=json.dumps(id_numero, sort_keys=True, separators=(",", ":")),
                   className="fila-var-nombre", title=e["descripcion"]),
        html.Div(dcc.Slider(id={"tipo": "sl", "var": c}, min=bajo, max=alto, step=paso, value=e["mediana"], marks=None, updatemode="mouseup"),
                 className="fila-var-deslizador", style={"backgroundImage": plots.huella_deslizador(vinos, c, bajo, alto)}),
        dbc.Input(id=id_numero, type="number", value=round(e["mediana"], 4), step="any", debounce=True, class_name="fila-var-numero"),
    ], className="fila-var")


def tab_prediccion():
    modelos = disponibles()
    if not modelos:
        return pendiente("La predicción en vivo")
    vinos = cargar_vinos()
    formulario = html.Section([
        html.Label("Modelo", htmlFor="sel-modelo", className="campo-etiqueta"),
        dbc.Select(id="sel-modelo", options=[{"label": m + (" (modelo original)" if m == ORIGINAL else ""), "value": m} for m in modelos],
                   value=modelos[0], class_name="mb-3"),
        html.Label("Tipo de vino", className="campo-etiqueta"),
        dbc.RadioItems(id="in-tipo", options=[{"label": html.Span([html.Span(className="gota gota-blanco"), "Blanco"]), "value": "white"},
                                              {"label": html.Span([html.Span(className="gota gota-tinto"), "Tinto"]), "value": "red"}],
                       value="white", inline=True, class_name="selector-tipo mb-3", input_class_name="chip-input", label_class_name="tipo-opcion",
                       label_checked_class_name="tipo-activo"),
        html.Div([html.Span("Cargar un ejemplo:", className="campo-etiqueta d-inline me-1"),
                  html.Button("Vino bueno real", id="ej-bueno", className="boton boton-suave", type="button"),
                  html.Button("Vino no bueno real", id="ej-malo", className="boton boton-suave", type="button"),
                  html.Button("Al azar", id="ej-azar", className="boton boton-suave", type="button"),
                  html.Button("Vino típico", id="ej-mediana", className="boton boton-suave", type="button",
                              title="Pone cada medición en su valor más habitual (la mediana)")],
                 className="d-flex flex-wrap gap-2 align-items-center mb-3"),
        html.P(["La sombra detrás de cada deslizador muestra dónde suelen estar los vinos buenos (azul) y los demás (ocre). ",
                "Arrastra o escribe el valor exacto."], className="small text-secondary"),
        html.Div([fila_variable(c, vinos) for c in NUMERICAS_VINO], className="ficha-lab"),
    ], className="tarjeta formulario-fijo")
    panel = html.Div([
        html.Div(dbc.Row([
            dbc.Col(html.Canvas(id="copa-pred", className="copa", role="img", title="Copa de la predicción",
                                **{"data-nivel": "0", "data-tipo": "white", "data-tono": "claro", "data-escala": "1"}), xs=6),
            dbc.Col([html.Div("Probabilidad de que sea bueno", className="prob-etiqueta"), html.Div("—", id="prob-valor", className="prob-valor"),
                     html.Div(id="pred-resumen")], xs=6, class_name="d-flex flex-column justify-content-center"),
        ], class_name="align-items-center"), className="panel-copa con-azulejo mb-3"),
        html.Div(id="pred-alerta"),
        dcc.Store(id="pred-datos"),
        dcc.Loading(html.Div(id="pred-explicacion"), type="dot", color="#1d5fc4", delay_show=400),
    ])
    lote = tarjeta("¿Tienes un lote completo?", [
        html.P("Sube un archivo CSV con una fila por vino y las columnas de la plantilla (la columna tipo_vino acepta «tinto» o «blanco»). El modelo "
               "elegido arriba ordena los vinos del más al menos prometedor, para que la cata empiece por los que más prometen."),
        html.Div([html.Button([html.I(className="bi bi-download me-2"), "Descargar la plantilla"], id="lote-plantilla-boton", type="button",
                              className="boton boton-suave"), dcc.Download(id="lote-plantilla")], className="mb-3"),
        dcc.Upload(id="lote-archivo", accept=".csv,text/csv", max_size=5_000_000,
                   children=html.Div([html.I(className="bi bi-cloud-arrow-up fs-3 d-block"), "Arrastra aquí tu CSV o haz clic para elegirlo"]),
                   className="area-carga"),
        dcc.Loading(html.Div(id="lote-resultado", className="mt-3"), type="dot", color="#1d5fc4"),
        dcc.Download(id="lote-descarga"),
        dcc.Store(id="lote-datos"),
    ])
    return html.Div([
        encabezado("¿Será bueno este vino?", etiqueta="2 · Predicción en vivo"),
        pasos("Carga un ejemplo o mueve las mediciones del vino.", "Mira la copa: se llena con la probabilidad de que obtenga 7 o más.",
              "Debajo de la copa está el porqué de esa respuesta."),
        dbc.Row([dbc.Col(formulario, lg=5, class_name="mb-4"), dbc.Col(panel, lg=7, class_name="mb-4")], class_name="align-items-start"),
        lote,
        ir_a("diagnostico", "Siguiente: ¿cuánto se equivoca?"),
    ])


def resumen_prediccion(ev):
    clase = "Vino bueno" if ev["clase"] == "bueno" else "Vino no bueno"
    icono = "bi-hand-thumbs-up" if ev["clase"] == "bueno" else "bi-hand-thumbs-down"
    fichas = [html.Span(c.capitalize(), className="ficha-clase" + (" encendida" if c in ev["conjunto"] else "")) for c in ("bueno", "no bueno")]
    if len(ev["conjunto"]) == 2:
        nota = "No está seguro: con 90 % de confianza no puede descartar ninguna de las dos respuestas."
    elif len(ev["conjunto"]) == 1:
        nota = f"Está seguro: con 90 % de confianza, la respuesta es «{ev['conjunto'][0]}»."
    else:
        nota = "Vino atípico para este modelo: ninguna respuesta alcanza el 90 % de confianza."
    cobertura = ev.get("cobertura")
    nota += (f" En la validación, esa garantía se cumplió en {100 * cobertura:.1f} % de los vinos." if cobertura is not None and np.isfinite(cobertura)
             else " La garantía nominal es del 90 %.")
    hijos = [html.Div([html.I(className=f"bi {icono}"), clase], className="prob-clase"),
             html.Div("Se marca «bueno» desde 50 %.", className="conforme-nota"),
             html.Div([html.Div(["¿Qué tan seguro está? (", termino("conjunto conforme"), " al 90 %)"], className="conforme-titulo"),
                       html.Div(fichas, className="conforme-fichas"), html.Div(nota, className="conforme-nota")], className="conforme")]
    if not ev["probabilistico"]:
        hijos.append(html.Div("SVM entrega una función de decisión; aquí se muestra transformada a la escala de 0 a 1.", className="conforme-nota"))
    if ev["provisional"]:
        hijos.append(html.Div([html.I(className="bi bi-exclamation-triangle"),
                               "Modelo provisional: el artefacto final se genera al terminar el cómputo pendiente. La predicción es real, pero "
                               "con menos árboles que el modelo final."], className="aviso-provisional"))
    return hijos


def balanza(exp):
    aportes = {a["variable"]: a for a in exp["aportes"]}
    orden = [a["variable"] for a in exp["aportes"]]
    tope = max(abs(a["valor"]) for a in exp["aportes"]) or 1
    filas = []
    for variable in NUMERICAS_VINO + ["tipo_vino"]:
        a = aportes.get(variable)
        if a is None:
            continue
        lado = "positiva" if a["valor"] > 0 else "negativa"
        filas.append(html.Div([
            html.Div(a["etiqueta"], className="balanza-nombre", title=a["etiqueta"]),
            html.Div([html.Div(className=f"balanza-barra {lado}"), html.Div(f"{a['valor']:+.3f}", className=f"balanza-valor {lado}")],
                     className="balanza-eje", style={"--v": f"{abs(a['valor']) / tope:.3f}"}),
        ], className="balanza-fila", key=variable, style={"--i": orden.index(variable)}))
    sube = [a for a in exp["aportes"] if a["valor"] > 0]
    baja = [a for a in exp["aportes"] if a["valor"] < 0]
    if sube and baja:
        titular = f"Lo que más le suma a este vino es su {minuscula(sube[0]['etiqueta'])}; lo que más le resta, su {minuscula(baja[0]['etiqueta'])}."
    elif sube:
        titular = f"Todo suma: lo que más pesa a favor es su {minuscula(sube[0]['etiqueta'])}."
    elif baja:
        titular = f"Todo resta: lo que más pesa en contra es su {minuscula(baja[0]['etiqueta'])}."
    else:
        titular = "Ninguna medición mueve la predicción."
    return tarjeta("¿Por qué piensa eso?", [
        como_leer("cada barra es una medición de este vino. Hacia la derecha (azul) sube la probabilidad de «bueno»; hacia la izquierda (ocre) la "
                  "baja. Cuanto más larga, más pesa."),
        html.Div([html.Div(), html.Div("← hacia «no bueno»"), html.Div("hacia «bueno» →")], className="balanza-polos"),
        html.Div(filas, className="balanza", style={"--filas": len(filas)}, role="list"),
        lectura(idea(titular, "Es una explicación ", termino("SHAP"), f": parte de {exp['base']:.3f}, lo que diría el modelo sin conocer el vino, y "
                     f"cada barra suma o resta desde ahí (en unidades de {exp['escala']}).")),
    ], clase="mb-3")


def explicacion_completa(exp, gem, pal, dist, error=None):
    hijos = []
    if exp is not None:
        hijos.append(balanza(exp))
    elif error:
        hijos.append(dbc.Alert(f"No se pudo calcular la explicación SHAP: {error}", color="warning"))
    teselas = [html.Div([str(n), html.Small("bueno" if n >= 7 else "no bueno")], className="gemelo " + ("bueno" if n >= 7 else "malo"),
                        title=f"Vino real con nota {n}") for n in gem["notas"]]
    total = len(gem["notas"])
    tarjeta_gemelos = tarjeta("¿Qué nota sacaron vinos parecidos?", [
        html.Div(teselas, className="gemelos"),
        lectura(idea(f"De los {total} vinos reales más parecidos, {gem['buenos']} {'fue bueno' if gem['buenos'] == 1 else 'fueron buenos'}.",
                     "Es una segunda opinión sencilla, sin modelo: si contradice a la copa, conviene catar el vino de todas formas.")),
    ])
    if pal:
        items = [html.Div([html.Div([html.Strong(f"{p['etiqueta']}: "), f"de {p['desde']:.3g} a {p['hasta']:.3g} {p['unidad']}"], className="palanca-texto"),
                           html.Div(f"{100 * p['p_desde']:.0f} % → {100 * p['p_hasta']:.0f} %", className="palanca-efecto")], className="palanca")
                 for p in pal]
        cuerpo_palancas = items + [lectura(idea("Así respondería el modelo si cambiara una sola medición.",
                                                "Las demás quedan igual y el cambio no se sale de lo habitual en los vinos. No es una receta: "
                                                "cambiar la química de un vino no garantiza mejor nota."))]
    else:
        cuerpo_palancas = [lectura(idea("Ningún cambio de una sola medición sube la probabilidad.",
                                        "Dentro de lo habitual en los vinos, el modelo ya ve este vino en su mejor versión."))]
    hijos.append(fila(tarjeta_gemelos, tarjeta("¿Qué lo haría subir?", cuerpo_palancas), clase="g-3 mb-3"))
    if dist is not None:
        niveles, prob = dist
        esperada = float(np.sum(niveles * prob))
        probable = int(niveles[int(np.argmax(prob))])
        azules = float(prob[niveles >= 7].sum())
        hijos.append(tarjeta(None, [plots.figura(plots.distribucion_nota(niveles, prob)),
                                    lectura(idea(f"La nota más probable para este vino es {probable}.",
                                                 f"Solo el modelo original predice la nota completa y no únicamente bueno / no bueno; en promedio "
                                                 f"espera un {esperada:.1f}. Las barras azules suman {100 * azules:.0f} %: es su estimación antes "
                                                 "del ajuste final de calibración, por eso puede diferir un poco de la copa."))]))
    return hijos


def tab_diagnostico():
    oof = tabla_app("oof_rep1.csv.gz")
    lideres = tabla_app("lideres.csv")
    if oof is None or lideres is None:
        return pendiente("El diagnóstico")
    opciones = [{"label": m + (" (modelo original)" if m == ORIGINAL else ""), "value": m}
                for m in lideres.sort_values("auc_media", ascending=False)["modelo"]]
    mosaico = tarjeta("¿Qué pasa con 100 vinos?", [
        html.Label(["¿Desde qué probabilidad se llama «bueno» a un vino? (", termino("umbral"), ")"], htmlFor="dec-umbral", className="campo-etiqueta"),
        dcc.Slider(id="dec-umbral", min=0.05, max=0.95, step=0.01, value=0.5, updatemode="drag",
                   marks={0.1: "10 %", 0.3: "30 %", 0.5: "50 %", 0.7: "70 %", 0.9: "90 %"}),
        html.Div(id="dec-umbral-texto", className="small text-secondary mb-2"),
        dbc.Row([
            dbc.Col(html.Div(id="mosaico", className="mosaico", role="img", **{"aria-label": "Cien vinos clasificados con el umbral elegido"}), sm=5),
            dbc.Col([
                html.Div("De cada 100 vinos", className="matriz-titulo"),
                html.Div([html.Div(), html.Div("El modelo dice «bueno»", className="rotulo"), html.Div("El modelo dice «no bueno»", className="rotulo"),
                          html.Div("En realidad es bueno", className="rotulo"),
                          html.Div([html.Strong(id="dec-vp"), "encontrados ✓"], className="celda"),
                          html.Div([html.Strong(id="dec-fn"), "perdidos ○"], className="celda"),
                          html.Div("En realidad no lo es", className="rotulo"),
                          html.Div([html.Strong(id="dec-fp"), "falsas alarmas !"], className="celda"),
                          html.Div([html.Strong(id="dec-vn"), "bien descartados"], className="celda")], className="matriz", role="table",
                         **{"aria-label": "Matriz de confusión por cada 100 vinos"}),
                html.Div(id="dec-detalle", className="small text-secondary mt-2"),
            ], sm=7),
        ]),
        html.Div([html.Div([html.Strong(id="dec-precision"), html.Span(["de los que marca como buenos lo son (", termino("precisión"), ")"])], className="metrica-mini"),
                  html.Div([html.Strong(id="dec-sensibilidad"), html.Span(["de los vinos buenos, encontrados (", termino("sensibilidad"), ")"])], className="metrica-mini"),
                  html.Div([html.Strong(id="dec-f1"), html.Span(["equilibrio entre las dos (", termino("F1"), ")"])], className="metrica-mini")],
                 className="metricas-decision"),
        html.Div([html.Span([html.I(style={"background": "#1d5fc4"}), "bueno encontrado ✓"]), html.Span([html.I(style={"boxShadow": "inset 0 0 0 2px #1d5fc4"}), "bueno perdido ○"]),
                  html.Span([html.I(style={"background": "#d07a10"}), "falsa alarma !"]), html.Span([html.I(style={"background": "#e8e1d2"}), "bien descartado"])],
                 className="leyenda mt-3"),
    ], clase="mb-3")
    costo = tarjeta("¿Qué umbral conviene?", [
        html.Label("¿Cuántas veces peor es perder un vino bueno que catar uno que no lo es?", htmlFor="dec-costo", className="campo-etiqueta"),
        dcc.Slider(id="dec-costo", min=1, max=10, step=1, value=3, updatemode="drag", marks={i: f"{i}×" for i in (1, 3, 5, 7, 10)}),
        lectura("", id="dec-sugerencia"),
    ])
    return html.Div([
        encabezado("¿Cuánto se equivoca?", "Todo se mide con vinos que el modelo no vio al entrenar. Mueve el umbral y mira cómo cambian los 100 "
                                           "vinos y el punto naranja de las curvas: bajar el umbral encuentra más vinos buenos, pero da más "
                                           "falsas alarmas.", etiqueta="3 · Diagnóstico"),
        dbc.Row(dbc.Col([html.Label("Modelo", htmlFor="diag-modelo", className="campo-etiqueta"),
                         dcc.Dropdown(id="diag-modelo", options=opciones, value=ORIGINAL, clearable=False)], md=5), class_name="mb-3"),
        dcc.Store(id="dec-datos", data=json_app("decision.json")),
        dbc.Row([
            dbc.Col([mosaico, costo], lg=6, class_name="mb-4"),
            dbc.Col([tarjeta(None, [plots.grafico("g-roc", 330),
                                    como_leer("cuanto más se pega la curva a la esquina de arriba a la izquierda, mejor. La diagonal es adivinar."),
                                    lectura("", id="l-roc")], clase="mb-3"),
                     tarjeta(None, [plots.grafico("g-pr", 330), lectura("", id="l-pr")])], lg=6, class_name="mb-4"),
        ]),
        fila(tarjeta(None, [plots.grafico("g-cal", 360),
                            como_leer("cada punto agrupa vinos con probabilidad parecida. Sobre la diagonal, el modelo dice justo lo que pasa; "
                                      "por debajo, promete de más."),
                            lectura("", id="l-cal")]), tarjeta(None, html.Div(id="diag-shap"))),
        ir_a("comparacion", "Siguiente: ¿la ventaja es real?"),
    ])


def contenido_shap(modelo):
    g = tabla_app("shap_global.csv")
    quien = "el modelo original" if modelo == ORIGINAL else modelo
    if g is not None and (g["modelo"] == modelo).any():
        d = g[g["modelo"] == modelo].copy()
        d["variable"] = d["variable"].map(lambda c: DICCIONARIO_VINO.get(c, (c,))[0])
        orden = d.sort_values("importancia", ascending=False)
        top, segundo = orden.iloc[0], orden.iloc[1]
        return [plots.figura(plots.importancia_global(d, f"¿En qué se fija más {quien}?")),
                lectura(idea(f"Se fija sobre todo en {minuscula(top['variable'])}.",
                             f"Pesa {top['importancia'] / segundo['importancia']:.1f} veces más que la segunda, {minuscula(segundo['variable'])}. "
                             "Es el promedio del peso ", termino("SHAP"), f" de cada medición en {int(d['n_vinos'].iloc[0])} vinos."))]
    e2 = tabla_app("e2_shap.csv")
    cuerpo = [pendiente(f"El resumen SHAP global de {modelo}")]
    if e2 is not None:
        d = e2.rename(columns={"clf_media_abs_shap": "importancia"})
        d = d.assign(variable=d["variable"].map(lambda c: DICCIONARIO_VINO.get(c, (c,))[0]))[["variable", "importancia"]]
        cuerpo = [plots.figura(plots.importancia_global(d, "SHAP global de Random Forest (Entregable 2)")),
                  lectura(["Mientras se generan los artefactos finales, este es el resumen ", termino("SHAP"),
                           " real del Random Forest del Entregable 2: el alcohol es, por lejos, la variable que más pesa."])]
    return cuerpo


def tab_comparacion():
    rangos = tabla_app("rangos_friedman.csv")
    pr = tabla_app("pruebas_bof.csv")
    v = json_app("veredicto.json")
    if rangos is None or pr is None or v is None:
        return pendiente("La comparación estadística")
    fr = json_app("friedman.json")
    serie = rangos.set_index(rangos.columns[0])["rango_medio"]
    bloques = tabla_app("bloques_auc.csv").set_index(["repeticion", "fold"])
    mejor = v["mejor_benchmark"]
    primero = serie.idxmin()
    empatan = [m for m in serie.sort_values().index if m != primero and serie[m] - serie[primero] <= fr["cd"]]
    quien = "El modelo original" if primero == ORIGINAL else primero
    titular_cd = (f"{quien} queda primero, pero empata con {', '.join(empatan[:-1]) + ' y ' + empatan[-1] if len(empatan) > 1 else empatan[0]}."
                  if empatan else f"{quien} queda primero y se separa de todos los demás.")
    lect_cd = idea(titular_cd, f"Su puesto promedio es {serie[primero]:.2f} y el de {mejor}, {serie[mejor]:.2f}. La prueba de ", termino("Friedman"),
                   f" confirma que los 10 modelos no rinden igual (F = {fr['F_ID']:.1f}, p {p_valor(fr['p_ID'])}), y la de ", termino("Nemenyi"),
                   f" no distingue a los que quedan dentro de la franja azul (a menos de {fr['cd']:.2f} puestos del primero).")
    formal = tabla_app("tabla_pruebas.csv")
    tabla_formal = tabla(
        [("Prueba", None), ("Hipótesis nula", "Lo que se supone cierto hasta que los datos digan lo contrario"), ("Estadístico", None),
         ("p crudo", "p-valor sin corregir"), ("p ajustado", "p-valor corregido por comparaciones múltiples. Menor que 0.05: diferencia real"),
         ("Efecto", "Tamaño de la diferencia, no solo si existe"), ("Conclusión", None)],
        [(legible(f.prueba), f.hipotesis_nula, f.estadistico, p_valor(f.p_crudo), p_valor(f.p_ajustado), f.efecto, legible(f.conclusion))
         for f in formal.itertuples()])
    columnas = [("benchmark", "Rival", "text", None), ("diferencia_media", "Ventaja media de AUC", "numeric", FORMATO_4),
                ("bloques_a_favor", "Exámenes ganados (de 25)", "numeric", None), ("wilcoxon_p_holm", "Wilcoxon (p ajustado)", "text", None),
                ("t_p_holm", "t corregida (p ajustado)", "text", None), ("delong_p_holm", "DeLong (p ajustado)", "text", None),
                ("delta_cliff", "δ de Cliff", "numeric", FORMATO_3), ("magnitud_cliff", "Tamaño del efecto", "text", None),
                ("bayes_p_mejor", "Prob. de que el original sea mejor", "numeric", FORMATO_3)]
    pr = pr.assign(**{c: pr[c].map(p_valor) for c in ("wilcoxon_p_holm", "t_p_holm", "delong_p_holm")})
    tabla_todos = dash_table.DataTable(data=pr.to_dict("records"), sort_action="native", page_action="none",
                                       columns=[{"id": c, "name": n, "type": ty, **({"format": fo} if fo else {})} for c, n, ty, fo in columnas],
                                       **ESTILO_TABLA)
    return html.Div([
        encabezado("¿La ventaja es real o es suerte?", "Los 10 modelos presentaron los mismos 25 exámenes. Cuando uno gana por muy poco, pudo "
                                                       "ser casualidad: las pruebas estadísticas dicen cuándo creerle a la diferencia.",
                   etiqueta="4 · Comparación estadística"),
        html.Div([html.I(className="bi bi-info-circle", **{"aria-hidden": "true"}),
                  html.Div(["La clave para leer esta pestaña es el ", termino("p-valor"), ": si es ", html.Strong("menor que 0.05"),
                            ", la diferencia difícilmente es casualidad; si es mayor, no hay evidencia suficiente para afirmar que un modelo es "
                            "mejor que el otro."])], className="clave"),
        fila(tarjeta(None, [plots.figura(plots.diagrama_cd(serie, fr["cd"], ORIGINAL, mejor)),
                            como_leer("más a la izquierda, mejor puesto promedio. Los modelos dentro de la franja azul no se distinguen del primero."),
                            lectura(lect_cd)]),
             tarjeta(None, [plots.figura(plots.cajas_bloques(bloques, ORIGINAL)),
                            lectura(idea("Las cajas de los primeros modelos se pisan entre sí.",
                                         "Cada punto es un examen. Un mismo modelo cambia más de un examen a otro que lo que lo separa del "
                                         "siguiente: por eso se comparan examen por examen, con pruebas pareadas."))]), clase="g-4 mb-4"),
        encabezado("El duelo, examen por examen", "Elige un rival. Cada baldosa es uno de los 25 exámenes (5 repeticiones × 5 pliegues) y dice por "
                                                  "cuánto ganó el modelo original (azul, con trofeo) o el rival (ocre, con ✕).",
                   etiqueta="Pruebas pareadas"),
        dbc.Row(dbc.Col([html.Label("Rival", htmlFor="duelo-rival", className="campo-etiqueta"),
                         dcc.Dropdown(id="duelo-rival", options=[{"label": b, "value": b} for b in pr["benchmark"]], value=mejor, clearable=False)],
                        md=5), class_name="mb-3"),
        dbc.Row([dbc.Col(tarjeta(None, html.Div(id="duelo")), lg=5, class_name="mb-4"),
                 dbc.Col(tarjeta(None, html.Div(id="jueces")), lg=7, class_name="mb-4")]),
        tarjeta(f"Tabla formal de pruebas frente al mejor rival ({mejor})",
                [html.P("Para quien quiera el detalle: cada fila es una prueba, con su p-valor antes y después del ajuste y el tamaño del efecto.",
                        className="small text-secondary"), tabla_formal], clase="mb-4"),
        tarjeta("Las mismas pruebas frente a cada uno de los 9 rivales",
                [html.P("Los p-valores están ajustados por Holm sobre las 9 comparaciones. Pulsa un encabezado para ordenar.",
                        className="small text-secondary"), tabla_todos]),
        ir_a("original", "Siguiente: el modelo original y su veredicto"),
    ])


def duelo(rival):
    b = tabla_app("bloques_auc.csv")
    dif = (b["Bosque Ordinal Focalizado"] - b[rival]).to_numpy()
    celdas = [html.Div()] + [html.Div(f"Pliegue {k + 1}", className="duelo-rotulo") for k in range(5)]
    for r in range(5):
        celdas.append(html.Div(f"Rep. {r + 1}", className="duelo-rotulo"))
        for k in range(5):
            i = r * 5 + k
            d = dif[i]
            celdas.append(html.Div([html.I(className="bi " + ("bi-trophy" if d > 0 else "bi-x-lg"), **{"aria-hidden": "true"}), f"{d:+.3f}"],
                                   className="duelo-baldosa " + ("gana" if d > 0 else "pierde"), style={"--i": i, "opacity": f"{0.55 + 0.45 * min(1, abs(d) / 0.02):.2f}"},
                                   title=f"Repetición {r + 1}, pliegue {k + 1}: diferencia de AUC {d:+.4f}"))
    gana = int((dif > 0).sum())
    return [html.H3(f"El modelo original gana {gana} de 25 exámenes", className="tarjeta-titulo"),
            html.Div(celdas, className="duelo", key=f"duelo-{rival}", role="img",
                     **{"aria-label": f"El modelo original gana {gana} de 25 exámenes frente a {rival}"}),
            lectura(idea("Ganar muchas veces por poco no es lo mismo que ganar por mucho.",
                         f"La ventaja media de AUC es {dif.mean():+.4f}. Las pruebas de al lado miden las dos cosas: cuántas veces gana y por cuánto."))]


def jueces(rival, mejor):
    pr = tabla_app("pruebas_bof.csv").set_index("benchmark").loc[rival]
    favorable = pr["diferencia_media"] > 0
    lista = [("Nemenyi", pr["nemenyi_p"], favorable, "compara puestos; la más exigente"),
             ("Wilcoxon", pr["wilcoxon_p_holm"], favorable, "compara examen por examen"),
             ("t corregida", pr["t_p_holm"], favorable, "la más prudente: los exámenes comparten vinos"),
             ("DeLong", pr["delong_p_holm"], pr["delong_dif"] > 0, "compara sobre los 5320 vinos")]
    tarjetas, votos = [], 0
    for i, (prueba, p, a_favor, papel) in enumerate(lista):
        rechaza = p < 0.05
        votos += int(rechaza and a_favor)
        texto, clase = (("A favor del original", "a-favor") if rechaza and a_favor else
                        ("A favor del rival", "en-contra") if rechaza else ("No ve diferencia", "en-contra"))
        tarjetas.append(html.Div([html.Div(termino(prueba), className="juez-nombre"), html.Div(papel, className="juez-papel"),
                                  html.Div(f"p ajustado = {p_valor(p)}", className="juez-p"),
                                  html.Div(texto, className=f"sello {clase}", style={"--i": i})], className="juez", key=f"{rival}-{prueba}"))
    if rival == mejor:
        regla = idea(f"{votos} de 4 pruebas ven una ventaja real; la regla exigía 3.",
                     f"Esa regla se fijó antes de los experimentos y se aplica frente al mejor rival, {mejor}.")
    else:
        regla = idea(f"{votos} de 4 pruebas ven una ventaja real del modelo original frente a {rival}.",
                     f"La regla de superioridad solo se aplica frente al mejor rival, {mejor}.")
    p_peor, p_eq, p_mejor = pr["bayes_p_peor"], pr["bayes_p_equivalente"], pr["bayes_p_mejor"]
    segmentos = [(p_peor, "#a85f08", "#fff", "el rival es mejor"), (p_eq, "#c9c2b3", "#10203f", "son equivalentes"),
                 (p_mejor, "#1d5fc4", "#fff", "el original es mejor")]
    barra = html.Div([html.Div(f"{100 * p:.0f} %" if p >= 0.09 else "", style={"flex": f"{max(p, 0.002)}", "background": fondo, "color": tinta},
                               title=f"Probabilidad de que {caso}: {100 * p:.0f} %") for p, fondo, tinta, caso in segmentos], className="segmentada mt-2")
    probable = max(segmentos, key=lambda s: s[0])
    return [html.H3("Las cuatro pruebas, como cuatro jueces", className="tarjeta-titulo"), html.Div(tarjetas, className="jueces"), lectura(regla),
            html.Div([html.Div([html.Strong(f"Lo más probable ({100 * probable[0]:.0f} %) es que {probable[3]}. "),
                                "Una quinta mirada, la prueba ", termino("t bayesiana"), ", reparte la probabilidad entre tres casos; "
                                "«equivalentes» significa menos de 0.005 de AUC de diferencia."], className="small mt-3"), barra,
                      html.Div([html.Span([html.I(style={"background": "#a85f08"}), "el rival es mejor"]),
                                html.Span([html.I(style={"background": "#c9c2b3"}), "son equivalentes"]),
                                html.Span([html.I(style={"background": "#1d5fc4"}), "el original es mejor"])], className="leyenda mt-2")])]
