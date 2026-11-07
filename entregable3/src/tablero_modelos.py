from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import numpy as np
from dash import dash_table, dcc, html

from . import plots
from .data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos, json_app, tabla_app
from .prediccion import ESQUEMA, ORIGINAL, disponibles
from .tablero import CORTOS, encabezado, fila, lectura, numero, pendiente, tarjeta, termino

BALANCEOS = {"sin_balanceo": "Sin balanceo", "class_weight": "Pesos por clase", "SMOTE": "SMOTE", "ADASYN": "ADASYN"}
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
)


def _prevalencia():
    d = json_app("decision.json")
    return d["positivos"] / (d["positivos"] + d["negativos"]) if d else 0.19


def _nan_a_nulo(valor):
    return None if valor is None or (isinstance(valor, float) and not np.isfinite(valor)) else valor


def tab_lideres():
    t = tabla_app("lideres.csv")
    if t is None:
        return pendiente("La tabla de líderes")
    filas = [{"modelo": r.modelo, "corto": CORTOS.get(r.modelo, r.modelo), "auc_media": _nan_a_nulo(r.auc_media),
              "pr_auc_media": _nan_a_nulo(r.pr_auc_media), "brier_media": _nan_a_nulo(r.brier_media), "destacado": r.modelo == ORIGINAL}
             for r in t.itertuples()]
    bof = t[t["modelo"] == ORIGINAL].iloc[0]
    mejor = t[t["modelo"] != ORIGINAL].sort_values("auc_media").iloc[-1]
    puesto = int(t.sort_values("auc_media", ascending=False)["modelo"].tolist().index(ORIGINAL)) + 1
    datos_tabla = t.assign(ic=t.apply(lambda f: f"[{f.auc_ic_inf:.3f}, {f.auc_ic_sup:.3f}]", axis=1))
    columnas = [("modelo", "Modelo", "text", None), ("balanceo", "Balanceo", "text", None), ("optimizador", "Optimizador", "text", None),
                ("auc_media", "AUC media", "numeric", FORMATO_4), ("auc_sd", "DE", "numeric", FORMATO_4), ("ic", "IC 95 % (bootstrap)", "text", None),
                ("pr_auc_media", "PR-AUC", "numeric", FORMATO_3), ("brier_media", "Brier", "numeric", FORMATO_3), ("f1_media", "F1 (0.5)", "numeric", FORMATO_3),
                ("prediccion_ms_por_caso", "Inferencia (ms/vino)", "numeric", FORMATO_3), ("tamano_mb", "Tamaño (MB)", "numeric", FORMATO_3)]
    tabla = dash_table.DataTable(
        id="tabla-lideres", data=datos_tabla.to_dict("records"), sort_action="native", filter_action="native", page_action="none",
        filter_options={"placeholder_text": "Filtrar…", "case": "insensitive"},
        columns=[{"id": c, "name": n, "type": ty, **({"format": fo} if fo else {})} for c, n, ty, fo in columnas],
        style_data_conditional=[{"if": {"filter_query": f'{{modelo}} = "{ORIGINAL}"'}, "backgroundColor": "#e4edf9", "fontWeight": 650}],
        sort_by=[{"column_id": "auc_media", "direction": "desc"}], **ESTILO_TABLA)
    lect = [f"El modelo original queda en el puesto {puesto} de 10 por AUC medio ({bof.auc_media:.4f} ± {bof.auc_sd:.4f}); el mejor rival es "
            f"{mejor.modelo} ({mejor.auc_media:.4f} ± {mejor.auc_sd:.4f}). Las diferencias entre los cuatro primeros son menores que la variación "
            "entre particiones (la DE): por eso la pestaña «Comparación» usa pruebas pareadas. Ordena o filtra la tabla (por ejemplo, escribe ",
            html.Code("> 0.85"), " bajo «AUC media»)."]
    if t["tamano_mb"].isna().all():
        lect.append(" El tamaño de cada artefacto aparece cuando se generen los modelos finales.")
    lentos = t.sort_values("prediccion_ms_por_caso")
    rapido, lento = lentos.iloc[0], lentos.iloc[-1]
    lect_costo = (f"Arriba a la izquierda está lo ideal: mucho AUC por poco tiempo. El más rápido es {rapido.modelo} ({rapido.prediccion_ms_por_caso:.3f} ms "
                  f"por vino) y el más lento {lento.modelo} ({lento.prediccion_ms_por_caso:.2f} ms). Aun el más lento evalúa unos "
                  f"{numero(1000 / lento.prediccion_ms_por_caso)} vinos por segundo: para catar un lote, el tiempo no es el cuello de botella.")
    return html.Div([
        encabezado("¿Qué modelo separa mejor los vinos buenos?",
                   ["Cada botella es un modelo con su mejor combinación de balanceo y optimizador, evaluada en 25 bloques (5 repeticiones × 5 pliegues). "
                    "Cuanto más llena, mejor; una botella vacía no supera al azar. Cambia la métrica y mira cómo se reordena la cava."],
                   etiqueta="La cava de los finalistas"),
        tarjeta(None, [
            dcc.Store(id="cava-datos", data={"filas": filas, "prevalencia": _prevalencia()}),
            dbc.RadioItems(id="cava-metrica", options=[{"label": "AUC", "value": "auc"}, {"label": "PR-AUC", "value": "pr"},
                                                       {"label": "Brier (menos es mejor)", "value": "brier"}],
                           value="auc", inline=True, class_name="cava-selector", input_class_name="chip-input", label_class_name="chip",
                           label_checked_class_name="chip-activo"),
            html.Div(id="cava", className="cava", role="list", **{"aria-label": "Modelos ordenados por la métrica elegida"}),
            html.Div([html.Span([html.I(style={"background": "#1d5fc4"}), "Bosque Ordinal Focalizado (modelo original)"]),
                      html.Span([html.I(style={"background": "#9aa3b2"}), "Benchmarks"])], className="leyenda cava-pie"),
            html.Div(["El número en la etiqueta de cada botella es su puesto. ¿Qué mide cada métrica? ", termino("AUC"), ", ", termino("PR-AUC"),
                      " y ", termino("Brier"), "."], className="small text-secondary mt-2"),
        ], clase="mb-4"),
        tarjeta("Tabla de líderes", [tabla, lectura(lect)], clase="mb-4"),
        fila(tarjeta(None, [plots.figura(plots.costo_exactitud(t, ORIGINAL, CORTOS)), lectura(lect_costo)]),
             tarjeta(None, torneo()), anchos=[5, 7]),
    ])


def torneo():
    clas = tabla_app("e2_clasificacion.csv")
    if clas is None:
        return pendiente("El torneo del Entregable 2")
    return [
        html.H3("¿Cómo se eligió a estos finalistas?", className="tarjeta-titulo"),
        html.P(["En el Entregable 2 compitieron 7 modelos con 4 técnicas de balanceo y 4 optimizadores. Cada baldosa es una combinación; la ★ marca la "
                "mejor de cada modelo, que es la que llega a la cava."], className="small text-secondary mb-2"),
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
        texto = (f"Como regresión, el mejor fue {mejor.modelo} con RMSE {mejor.rmse:.3f} puntos de nota. La distancia entre el mejor y el peor modelo "
                 f"({rango_modelos.max() - rango_modelos.min():.3f}) es mucho mayor que la que separa a los optimizadores de un mismo modelo.")
        return fig, texto
    clas = tabla_app("e2_clasificacion.csv")
    clas = clas.assign(balanceo=clas["balanceo"].map(BALANCEOS).fillna(clas["balanceo"]),
                       optimizador=clas["optimizador"].map(OPTIMIZADORES).fillna(clas["optimizador"]))
    fig = plots.mural(clas, "modelo", ["balanceo", "optimizador"], "auc", "Torneo de clasificación: AUC de cada combinación", "AUC")
    entre = clas.groupby("modelo")["auc"].mean()
    dentro = clas.groupby("modelo")["auc"].agg(lambda s: s.max() - s.min()).mean()
    mejor = clas.sort_values("auc").iloc[-1]
    texto = (f"Las filas cambian mucho de color y las columnas casi nada: entre el mejor y el peor modelo hay {entre.max() - entre.min():.3f} de AUC, "
             f"pero cambiar el balanceo o el optimizador de un mismo modelo mueve en promedio {dentro:.3f}. El modelo importa más que el ajuste. "
             f"La mejor combinación fue {mejor.modelo} con {mejor.balanceo.lower()} y {mejor.optimizador} (AUC {mejor.auc:.4f}).")
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
        html.Div([html.Span("Cargar:", className="campo-etiqueta d-inline me-1"),
                  html.Button("Vino bueno real", id="ej-bueno", className="boton boton-suave", type="button"),
                  html.Button("Vino no bueno real", id="ej-malo", className="boton boton-suave", type="button"),
                  html.Button("Al azar", id="ej-azar", className="boton boton-suave", type="button"),
                  html.Button("Medianas", id="ej-mediana", className="boton boton-suave", type="button")],
                 className="d-flex flex-wrap gap-2 align-items-center mb-3"),
        html.P(["La silueta detrás de cada deslizador muestra dónde están los vinos buenos (azul) y los no buenos (ocre). ",
                "Puedes arrastrar o escribir el valor exacto."], className="small text-secondary"),
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
        encabezado("¿Será bueno este vino?", ["Mueve los deslizadores o escribe los valores de laboratorio: la copa se llena con la probabilidad de que el "
                                              "vino obtenga 7 o más, y abajo aparece por qué el modelo piensa eso."],
                   etiqueta="Predicción en vivo"),
        dbc.Row([dbc.Col(formulario, lg=5, class_name="mb-4"), dbc.Col(panel, lg=7, class_name="mb-4")]),
        lote,
    ])


def resumen_prediccion(ev):
    clase = "Vino bueno" if ev["clase"] == "bueno" else "Vino no bueno"
    icono = "bi-hand-thumbs-up" if ev["clase"] == "bueno" else "bi-hand-thumbs-down"
    fichas = [html.Span(c.capitalize(), className="ficha-clase" + (" encendida" if c in ev["conjunto"] else "")) for c in ("bueno", "no bueno")]
    if len(ev["conjunto"]) == 2:
        nota = "Las dos respuestas son compatibles: el modelo no tiene evidencia suficiente para decidir con 90 % de garantía."
    elif len(ev["conjunto"]) == 1:
        nota = f"Con 90 % de garantía, la respuesta es «{ev['conjunto'][0]}»."
    else:
        nota = "Ninguna respuesta alcanza la garantía del 90 %: el vino es atípico para este modelo."
    cobertura = ev.get("cobertura")
    nota += (f" Cobertura empírica medida: {100 * cobertura:.1f} %." if cobertura is not None and np.isfinite(cobertura)
             else " Cobertura nominal del 90 %; la cobertura empírica se mide en los estudios de robustez.")
    hijos = [html.Div([html.I(className=f"bi {icono}"), clase], className="prob-clase"),
             html.Div([html.Div([termino("Conjunto conforme", "conjunto conforme"), " al 90 %"], className="conforme-titulo"),
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
    partes = []
    if sube:
        partes.append(f"Lo que más empuja hacia «bueno» es {sube[0]['etiqueta'].lower()} ({sube[0]['valor']:+.3f})")
    if baja:
        partes.append(f"lo que más frena es {baja[0]['etiqueta'].lower()} ({baja[0]['valor']:+.3f})")
    texto = (", y ".join(partes) + ". " if partes else "") + (f"El valor base ({exp['base']:.3f}) es lo que diría el modelo sin conocer el vino; "
                                                               f"cada barra suma o resta desde ahí, en unidades de {exp['escala']}.")
    return tarjeta("¿Por qué piensa eso?", [
        html.Div([html.Div(), html.Div("← hacia «no bueno»"), html.Div("hacia «bueno» →")], className="balanza-polos"),
        html.Div(filas, className="balanza", style={"--filas": len(filas)}, role="list"),
        lectura([termino("SHAP"), ": " + texto[0].lower() + texto[1:]] if texto else ""),
    ], clase="mb-3")


def explicacion_completa(exp, gem, pal, dist, error=None):
    hijos = []
    if exp is not None:
        hijos.append(balanza(exp))
    elif error:
        hijos.append(dbc.Alert(f"No se pudo calcular la explicación SHAP: {error}", color="warning"))
    teselas = [html.Div([str(n), html.Small("bueno" if n >= 7 else "no bueno")], className="gemelo " + ("bueno" if n >= 7 else "malo"),
                        title=f"Vino real con nota {n}") for n in gem["notas"]]
    tarjeta_gemelos = tarjeta("¿Qué notas sacaron vinos parecidos?", [
        html.Div(teselas, className="gemelos"),
        lectura(f"De los {len(gem['notas'])} vinos reales más parecidos del mismo tipo, {gem['buenos']} fueron buenos. Es una segunda opinión "
                "sencilla, sin modelo: si contradice a la copa, conviene catar el vino de todas formas."),
    ])
    if pal:
        items = [html.Div([html.Div([html.Strong(f"{p['etiqueta']}: "), f"de {p['desde']:.3g} a {p['hasta']:.3g} {p['unidad']}"], className="palanca-texto"),
                           html.Div(f"{100 * p['p_desde']:.0f} % → {100 * p['p_hasta']:.0f} %", className="palanca-efecto")], className="palanca")
                 for p in pal]
        cuerpo_palancas = items + [lectura("Cada fila cambia una sola variable dentro del rango habitual de los vinos y deja las demás igual. Describe "
                                           "cómo responde el modelo, no una receta: cambiar la química de un vino no garantiza mejor nota.")]
    else:
        cuerpo_palancas = [lectura("Ningún cambio de una sola variable, dentro del rango habitual, sube la probabilidad: el modelo ya ve este vino en su "
                                   "mejor versión posible.")]
    hijos.append(fila(tarjeta_gemelos, tarjeta("¿Qué lo haría subir?", cuerpo_palancas), clase="g-3 mb-3"))
    if dist is not None:
        niveles, prob = dist
        esperada = float(np.sum(niveles * prob))
        azules = float(prob[niveles >= 7].sum())
        hijos.append(tarjeta(None, [plots.figura(plots.distribucion_nota(niveles, prob)),
                                    lectura(f"El modelo original predice la nota completa, no solo bueno / no bueno. La nota esperada es {esperada:.1f}. "
                                            f"Las barras azules suman {100 * azules:.0f} %: es la estimación directa de los árboles. La copa muestra la "
                                            "probabilidad ya calibrada con los vinos fuera de bolsa, por eso las dos cifras pueden diferir un poco.")]))
    return hijos


def tab_diagnostico():
    oof = tabla_app("oof_rep1.csv.gz")
    lideres = tabla_app("lideres.csv")
    if oof is None or lideres is None:
        return pendiente("El diagnóstico")
    opciones = [{"label": m, "value": m} for m in lideres.sort_values("auc_media", ascending=False)["modelo"]]
    mosaico = tarjeta("¿Qué pasa con 100 vinos?", [
        html.Label("Umbral de decisión", htmlFor="dec-umbral", className="campo-etiqueta"),
        dcc.Slider(id="dec-umbral", min=0.05, max=0.95, step=0.01, value=0.5, updatemode="drag",
                   marks={0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.7: "0.7", 0.9: "0.9"}),
        html.Div(id="dec-umbral-texto", className="small text-secondary mb-2"),
        dbc.Row([
            dbc.Col(html.Div(id="mosaico", className="mosaico", role="img", **{"aria-label": "Cien vinos clasificados con el umbral elegido"}), sm=6),
            dbc.Col([
                html.Div([html.Div(), html.Div("Predicho bueno", className="rotulo"), html.Div("Predicho no bueno", className="rotulo"),
                          html.Div("Real bueno", className="rotulo"),
                          html.Div([html.Strong(id="dec-vp"), "encontrados ✓"], className="celda"),
                          html.Div([html.Strong(id="dec-fn"), "perdidos ○"], className="celda"),
                          html.Div("Real no bueno", className="rotulo"),
                          html.Div([html.Strong(id="dec-fp"), "falsas alarmas !"], className="celda"),
                          html.Div([html.Strong(id="dec-vn"), "bien descartados"], className="celda")], className="matriz"),
                html.Div([html.Div([html.Strong(id="dec-precision"), termino("Precisión", "precisión")], className="metrica-mini"),
                          html.Div([html.Strong(id="dec-sensibilidad"), termino("Sensibilidad", "sensibilidad")], className="metrica-mini"),
                          html.Div([html.Strong(id="dec-f1"), termino("F1")], className="metrica-mini")], className="metricas-decision"),
            ], sm=6),
        ]),
        html.Div([html.Span([html.I(style={"background": "#1d5fc4"}), "bueno encontrado"]), html.Span([html.I(style={"boxShadow": "inset 0 0 0 2px #1d5fc4"}), "bueno perdido"]),
                  html.Span([html.I(style={"background": "#d07a10"}), "falsa alarma"]), html.Span([html.I(style={"background": "#e8e1d2"}), "bien descartado"])],
                 className="leyenda mt-3"),
        html.Hr(),
        html.Label("¿Cuánto cuesta perder un vino bueno frente a catar uno que no lo es?", htmlFor="dec-costo", className="campo-etiqueta"),
        dcc.Slider(id="dec-costo", min=1, max=10, step=1, value=3, updatemode="drag", marks={i: f"{i}×" for i in (1, 3, 5, 7, 10)}),
        lectura("", id="dec-sugerencia"),
    ])
    return html.Div([
        encabezado("¿Puedo confiar en él?", ["Todo se calcula con predicciones fuera de muestra: cada vino lo predijo un modelo que no lo vio al entrenar. "
                                             "Mueve el umbral y mira cómo cambian los 100 vinos, la matriz de confusión y el punto naranja de las curvas."],
                   etiqueta="Diagnóstico"),
        dbc.Row(dbc.Col([html.Label("Modelo", htmlFor="diag-modelo", className="campo-etiqueta"),
                         dcc.Dropdown(id="diag-modelo", options=opciones, value=ORIGINAL, clearable=False)], md=5), class_name="mb-3"),
        dcc.Store(id="dec-datos", data=json_app("decision.json")),
        dbc.Row([
            dbc.Col(mosaico, lg=6, class_name="mb-4"),
            dbc.Col([tarjeta(None, [plots.grafico("g-roc", 330), lectura("", id="l-roc")], clase="mb-3"),
                     tarjeta(None, [plots.grafico("g-pr", 330), lectura("", id="l-pr")])], lg=6, class_name="mb-4"),
        ]),
        fila(tarjeta(None, [plots.grafico("g-cal", 360), lectura("", id="l-cal")]), tarjeta(None, html.Div(id="diag-shap"))),
    ])


def contenido_shap(modelo):
    g = tabla_app("shap_global.csv")
    if g is not None and (g["modelo"] == modelo).any():
        d = g[g["modelo"] == modelo].copy()
        d["variable"] = d["variable"].map(lambda c: DICCIONARIO_VINO.get(c, (c,))[0])
        top = d.sort_values("importancia").iloc[-1]
        return [plots.figura(plots.importancia_global(d, f"¿Qué variables usa más {CORTOS.get(modelo, modelo)}?")),
                lectura([f"La variable que más mueve sus predicciones es {top['variable'].lower()}. Promedio del valor absoluto del aporte ",
                         termino("SHAP"), f" sobre {int(d['n_vinos'].iloc[0])} vinos."])]
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
    lect_cd = [termino("Friedman"), f": F = {fr['F_ID']:.1f}, p = {fr['p_ID']:.1g}; los modelos no son iguales. El modelo original tiene el mejor "
               f"rango medio ({serie[ORIGINAL]:.2f}) y {mejor} le sigue ({serie[mejor]:.2f}). Todos los que caen dentro de la franja azul (CD = {fr['cd']:.2f}) "
               "no se distinguen del primero según ", termino("Nemenyi"), "."]
    formal = tabla_app("tabla_pruebas.csv")
    p_txt = lambda x: "< 0.0001" if x < 1e-4 else f"{x:.4f}"
    tabla_formal = dbc.Table([html.Thead(html.Tr([html.Th(c) for c in ("Prueba", "Hipótesis nula", "Estadístico", "p crudo", "p ajustado", "Efecto", "Conclusión")])),
                              html.Tbody([html.Tr([html.Td(f.prueba), html.Td(f.hipotesis_nula), html.Td(f.estadistico), html.Td(p_txt(f.p_crudo)),
                                                   html.Td(p_txt(f.p_ajustado)), html.Td(f.efecto), html.Td(f.conclusion)]) for f in formal.itertuples()])],
                             size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")
    columnas = [("benchmark", "Rival", "text", None), ("diferencia_media", "ΔAUC medio", "numeric", FORMATO_4),
                ("bloques_a_favor", "Bloques a favor (de 25)", "numeric", None), ("wilcoxon_p_holm", "Wilcoxon (Holm)", "text", None),
                ("t_p_holm", "t corregida (Holm)", "text", None), ("delong_p_holm", "DeLong (Holm)", "text", None),
                ("delta_cliff", "δ de Cliff", "numeric", FORMATO_3), ("magnitud_cliff", "Magnitud", "text", None),
                ("bayes_p_mejor", "P(BOF mejor) bayesiana", "numeric", FORMATO_3)]
    pr = pr.assign(**{c: pr[c].map(p_txt) for c in ("wilcoxon_p_holm", "t_p_holm", "delong_p_holm")})
    tabla_todos = dash_table.DataTable(data=pr.to_dict("records"), sort_action="native", page_action="none",
                                       columns=[{"id": c, "name": n, "type": ty, **({"format": fo} if fo else {})} for c, n, ty, fo in columnas],
                                       **ESTILO_TABLA)
    return html.Div([
        encabezado("¿La diferencia es real o es suerte?", ["Los 10 modelos compitieron en los mismos 25 bloques. Una diferencia pequeña en el AUC medio "
                                                           "puede deberse al azar de las particiones: estas pruebas lo separan."],
                   etiqueta="Comparación estadística"),
        fila(tarjeta(None, [plots.figura(plots.diagrama_cd(serie, fr["cd"], ORIGINAL, mejor)), lectura(lect_cd)]),
             tarjeta(None, [plots.figura(plots.cajas_bloques(bloques, ORIGINAL)),
                            lectura("Cada punto es un bloque. La dispersión dentro de cada modelo es mayor que muchas de las diferencias entre modelos: "
                                    "por eso se comparan bloque a bloque, con pruebas pareadas.")]), clase="g-4 mb-4"),
        encabezado("El duelo, bloque por bloque", "Elige un rival: cada baldosa es un bloque y dice por cuánto ganó el modelo original (azul) o el rival (ocre).",
                   etiqueta="Pruebas pareadas"),
        dbc.Row(dbc.Col([html.Label("Rival", htmlFor="duelo-rival", className="campo-etiqueta"),
                         dcc.Dropdown(id="duelo-rival", options=[{"label": b, "value": b} for b in pr["benchmark"]], value=mejor, clearable=False)],
                        md=5), class_name="mb-3"),
        dbc.Row([dbc.Col(tarjeta(None, html.Div(id="duelo")), lg=5, class_name="mb-4"),
                 dbc.Col(tarjeta(None, html.Div(id="jueces")), lg=7, class_name="mb-4")]),
        tarjeta(f"Tabla de pruebas frente al mejor rival ({mejor})", [tabla_formal], clase="mb-4"),
        tarjeta("Pruebas frente a cada rival (Holm sobre 9 comparaciones)", [tabla_todos]),
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
    return [html.H3(f"{gana} de 25 bloques para el modelo original", className="tarjeta-titulo"),
            html.Div(celdas, className="duelo", key=f"duelo-{rival}", role="img",
                     **{"aria-label": f"El modelo original gana {gana} de 25 bloques frente a {rival}"}),
            lectura(f"Diferencia media de AUC: {dif.mean():+.4f}. Ganar muchos bloques por poco no es lo mismo que ganar por mucho: las pruebas de al "
                    "lado miden ambas cosas.")]


def jueces(rival, mejor):
    pr = tabla_app("pruebas_bof.csv").set_index("benchmark").loc[rival]
    favorable = pr["diferencia_media"] > 0
    lista = [("Nemenyi", pr["nemenyi_p"], favorable, "Nemenyi"), ("Wilcoxon", pr["wilcoxon_p_holm"], favorable, "Wilcoxon"),
             ("t corregida", pr["t_p_holm"], favorable, "t corregida"), ("DeLong", pr["delong_p_holm"], pr["delong_dif"] > 0, "DeLong")]
    tarjetas, votos = [], 0
    for i, (nombre, p, a_favor, clave) in enumerate(lista):
        rechaza = p < 0.05
        votos += int(rechaza and a_favor)
        texto, clase = (("Diferencia real a favor del BOF", "a-favor") if rechaza and a_favor else
                        ("Diferencia real a favor del rival", "en-contra") if rechaza else ("No ve diferencia", "en-contra"))
        tarjetas.append(html.Div([html.Div(termino(nombre, clave), className="juez-nombre"),
                                  html.Div(f"p ajustado = {'< 0.0001' if p < 1e-4 else f'{p:.4f}'}", className="juez-p"),
                                  html.Div(texto, className=f"sello {clase}", style={"--i": i})], className="juez", key=f"{rival}-{nombre}"))
    regla = (f"La regla pre-registrada exige que al menos 3 de los 4 jueces vean una diferencia a favor frente al mejor rival ({mejor}). "
             f"Aquí lo hacen {votos}.") if rival == mejor else (f"{votos} de 4 jueces ven una diferencia a favor del modelo original frente a {rival}. "
                                                                   f"La regla de superioridad se aplica frente al mejor rival, {mejor}.")
    p_peor, p_eq, p_mejor = pr["bayes_p_peor"], pr["bayes_p_equivalente"], pr["bayes_p_mejor"]
    segmentos = [(p_peor, "#a85f08", "#fff", "rival mejor"), (p_eq, "#c9c2b3", "#10203f", "equivalentes"), (p_mejor, "#1d5fc4", "#fff", "BOF mejor")]
    barra = html.Div([html.Div(f"{100 * p:.0f} %" if p >= 0.09 else "", style={"flex": f"{max(p, 0.002)}", "background": fondo, "color": tinta},
                               title=f"P({nombre}) = {p:.2f}") for p, fondo, tinta, nombre in segmentos], className="segmentada mt-2")
    return [html.H3("Los cuatro jueces", className="tarjeta-titulo"), html.Div(tarjetas, className="jueces"), lectura(regla),
            html.Div([html.Div(["Prueba ", termino("t bayesiana"), " correlacionada: probabilidad de que cada uno sea mejor por más de 0.005 de AUC"],
                               className="small text-secondary mt-3"), barra,
                      html.Div([html.Span([html.I(style={"background": "#a85f08"}), "rival mejor"]), html.Span([html.I(style={"background": "#c9c2b3"}), "equivalentes"]),
                                html.Span([html.I(style={"background": "#1d5fc4"}), "BOF mejor"])], className="leyenda mt-2")])]
