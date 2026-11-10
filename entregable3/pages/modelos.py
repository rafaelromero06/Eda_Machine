import base64
import io
from urllib.parse import parse_qs

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, MATCH, ClientsideFunction, Input, Output, State, callback, clientside_callback, dcc, html, no_update
from sklearn.metrics import precision_recall_curve, roc_curve

from src import plots
from src.data import CATEGORICAS_VINO, NUMERICAS_VINO, cargar_vinos, json_app, tabla_app
from src.original_model import auc_mann_whitney
from src.prediccion import (COLUMNAS, ESQUEMA, armar_vino, distribucion_nota, evaluar, evaluar_lote, explicar, gemelos, palancas,
                            validar)
from src.tablero import idea, siguiente, termino
from src.tablero_modelos import (contenido_shap, duelo, explicacion_completa, figura_torneo, jueces, resumen_prediccion, tab_comparacion,
                                 tab_diagnostico, tab_lideres, tab_prediccion)

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Cata Ciega", order=2)

VINOS = cargar_vinos()
FONDO = VINOS.sample(200, random_state=0)[NUMERICAS_VINO + CATEGORICAS_VINO]
PESTANAS = {"lideres": "1 · Tabla de líderes", "prediccion": "2 · Predicción en vivo", "diagnostico": "3 · Diagnóstico",
            "comparacion": "4 · Comparación estadística", "original": "5 · Modelo original"}


def tab_original():
    from src.tablero_original import contenido
    return contenido()


layout = html.Div([
    dcc.Location(id="url-modelos", refresh=False),
    html.Section(dbc.Container([
        html.Div("Modelos", className="antetitulo"),
        html.H1("De la química a la copa"),
        html.P("Cinco pestañas, en orden: quién acierta más, cómo se predice un vino, cuánto se equivoca cada modelo, si la ventaja es real "
               "y cómo funciona el modelo original."),
    ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
    html.Section(dbc.Container([
        dbc.Tabs([dbc.Tab(label=texto, tab_id=clave) for clave, texto in PESTANAS.items()], id="pestanas", active_tab="lideres", class_name="pestanas"),
        dcc.Loading(html.Div(id="contenido-pestana"), type="dot", color="#1d5fc4", delay_show=300),
    ], fluid="xl"), className="banda banda-compacta"),
    siguiente("/modelos", "¿Le ganarías al modelo? En la cata a ciegas eliges entre dos vinos reales y comparas tu acierto con el suyo."),
])


@callback(Output("pestanas", "active_tab"), Input("url-modelos", "search"))
def pestana_desde_url(busqueda):
    pedida = parse_qs((busqueda or "").lstrip("?")).get("pestana", ["lideres"])[0]
    return pedida if pedida in PESTANAS else "lideres"


@callback(Output("contenido-pestana", "children"), Input("pestanas", "active_tab"))
def renderizar(pestana):
    return {"lideres": tab_lideres, "prediccion": tab_prediccion, "diagnostico": tab_diagnostico, "comparacion": tab_comparacion,
            "original": tab_original}.get(pestana, tab_lideres)()


clientside_callback(ClientsideFunction("cata", "ir"), Output("pestanas", "active_tab", allow_duplicate=True),
                    Input({"tipo": "ir", "pestana": ALL}, "n_clicks"), prevent_initial_call=True)

clientside_callback(ClientsideFunction("cata", "recordar"), Output("contenido-pestana", "title"), Input("pestanas", "active_tab"))

clientside_callback(ClientsideFunction("cata", "cava"), Output("cava", "title"), Output("cava-explica", "children"),
                    Input("cava-datos", "data"), Input("cava-metrica", "value"))


@callback(Output("g-torneo", "figure"), Output("l-torneo", "children"), Input("torneo-tarea", "value"))
def actualizar_torneo(tarea):
    return figura_torneo(tarea)


clientside_callback(ClientsideFunction("cata", "sincronizar"),
                    Output({"tipo": "sl", "var": MATCH}, "value"), Output({"tipo": "num", "var": MATCH}, "value"),
                    Input({"tipo": "num", "var": MATCH}, "value"), Input({"tipo": "sl", "var": MATCH}, "value"), prevent_initial_call=True)


@callback(Output({"tipo": "num", "var": ALL}, "value", allow_duplicate=True), Output({"tipo": "sl", "var": ALL}, "value", allow_duplicate=True),
          Output("in-tipo", "value"),
          Input("ej-bueno", "n_clicks"), Input("ej-malo", "n_clicks"), Input("ej-azar", "n_clicks"), Input("ej-mediana", "n_clicks"),
          prevent_initial_call=True)
def cargar_ejemplo(*_):
    quien = dash.ctx.triggered_id
    if quien == "ej-mediana":
        valores, tipo = [round(ESQUEMA[c]["mediana"], 4) for c in NUMERICAS_VINO], "white"
    else:
        sub = {"ej-bueno": VINOS[VINOS["quality"] >= 7], "ej-malo": VINOS[VINOS["quality"] <= 5]}.get(quien, VINOS)
        fila = sub.iloc[int(np.random.default_rng().integers(len(sub)))]
        valores, tipo = [float(fila[c]) for c in NUMERICAS_VINO], fila["tipo_vino"]
    return valores, valores, tipo


@callback(Output("pred-datos", "data"), Output("pred-resumen", "children"), Output("pred-alerta", "children"),
          Input({"tipo": "num", "var": ALL}, "value"), Input("in-tipo", "value"), Input("sel-modelo", "value"))
def predecir(valores, tipo, modelo):
    errores = validar(valores)
    if errores:
        return no_update, no_update, dbc.Alert([html.H6("Revisa los datos del vino", className="alert-heading"),
                                                html.Ul([html.Li(e) for e in errores], className="mb-0")], color="danger")
    try:
        ev = evaluar(modelo, armar_vino(valores, tipo))
    except Exception as exc:
        return no_update, no_update, dbc.Alert(f"No se pudo calcular la predicción: {exc}", color="danger")
    return {**ev, "tipo": tipo, "valores": [float(v) for v in valores]}, resumen_prediccion(ev), None


clientside_callback(ClientsideFunction("cata", "copa"), Output("copa-pred", "title"), Input("pred-datos", "data"))


@callback(Output("pred-explicacion", "children"), Input("pred-datos", "data"))
def explicar_prediccion(datos):
    if not datos:
        return no_update
    X = armar_vino(datos["valores"], datos["tipo"])
    exp, error = None, None
    try:
        exp = explicar(datos["modelo"], X, FONDO)
    except Exception as exc:
        error = str(exc)
    return explicacion_completa(exp, gemelos(datos["valores"], datos["tipo"]), palancas(datos["modelo"], X), distribucion_nota(datos["modelo"], X), error)


@callback(Output("lote-plantilla", "data"), Input("lote-plantilla-boton", "n_clicks"), prevent_initial_call=True)
def descargar_plantilla(_):
    ejemplo = VINOS.sample(5, random_state=7)[COLUMNAS].assign(tipo_vino=lambda d: d["tipo_vino"].map({"red": "tinto", "white": "blanco"}))
    return dcc.send_data_frame(ejemplo.to_csv, "plantilla_vinos.csv", index=False)


@callback(Output("lote-resultado", "children"), Output("lote-datos", "data"),
          Input("lote-archivo", "contents"), State("lote-archivo", "filename"), State("sel-modelo", "value"), prevent_initial_call=True)
def procesar_lote(contenido, nombre, modelo):
    try:
        crudo = base64.b64decode(contenido.split(",", 1)[1]).decode("utf-8-sig")
        tabla = pd.read_csv(io.StringIO(crudo), sep=None, engine="python")
    except Exception:
        return dbc.Alert("No se pudo leer el archivo. Debe ser un CSV de texto, separado por comas o punto y coma.", color="danger"), None
    resultado, errores = evaluar_lote(modelo, tabla)
    avisos = [dbc.Alert([html.H6("Algunas filas no se evaluaron", className="alert-heading"), html.Ul([html.Li(e) for e in errores], className="mb-0")],
                        color="warning")] if errores else []
    if resultado is None:
        return avisos or dbc.Alert("No hay filas válidas para evaluar.", color="danger"), None
    top = resultado.head(10)
    tabla_html = dbc.Table([html.Thead(html.Tr([html.Th("Puesto"), html.Th("Fila del archivo"), html.Th("Probabilidad de ser bueno"), html.Th("Tipo"),
                                                html.Th("Alcohol")])),
                            html.Tbody([html.Tr([html.Td(i + 1), html.Td(int(f.fila)), html.Td(f"{100 * f.probabilidad_bueno:.0f} %"),
                                                 html.Td("Tinto" if f.tipo_vino == "red" else "Blanco"), html.Td(f"{f.alcohol:.1f}")])
                                        for i, f in enumerate(top.itertuples())])], size="sm", hover=True, class_name="tabla-numeros")
    esperados = resultado["probabilidad_bueno"].sum()
    return avisos + [
        html.P(f"«{nombre}»: {len(resultado)} vinos evaluados con {modelo}. Según el modelo, cabe esperar unos {esperados:.0f} vinos buenos en el lote; "
               "estos son los 10 que conviene catar primero:"),
        tabla_html,
        html.Button([html.I(className="bi bi-download me-2"), "Descargar el lote ordenado"], id="lote-descarga-boton", type="button", className="boton"),
    ], resultado.to_dict("records")


@callback(Output("lote-descarga", "data"), Input("lote-descarga-boton", "n_clicks"), State("lote-datos", "data"), prevent_initial_call=True)
def descargar_lote(_, datos):
    if not datos:
        return no_update
    return dcc.send_data_frame(pd.DataFrame(datos).to_csv, "lote_ordenado.csv", index=False)


@callback(Output("g-roc", "figure"), Output("l-roc", "children"), Output("g-pr", "figure"), Output("l-pr", "children"),
          Output("g-cal", "figure"), Output("l-cal", "children"), Output("diag-shap", "children"), Input("diag-modelo", "value"))
def diagnostico(modelo):
    oof = tabla_app("oof_rep1.csv.gz")
    y, s = oof["y_evento"].to_numpy(), oof[modelo].to_numpy()
    p = s if modelo != "SVM" else 1 / (1 + np.exp(-s))
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc = auc_mann_whitney(y, p)
    pr_auc = float(np.sum(np.diff(rec[::-1]) * prec[::-1][1:]))
    l_roc = idea(f"De cada 100 parejas de vinos, ordena bien {100 * auc:.0f}.",
                 "Una pareja es un vino bueno y uno que no lo es, tomados al azar; acierta si le da más puntaje al bueno. Eso mide el ", termino("AUC"),
                 f" ({auc:.3f}). El punto naranja es el umbral elegido a la izquierda.")
    l_pr = idea("Encontrar más vinos buenos cuesta más falsas alarmas.",
                f"Al avanzar hacia la derecha el modelo encuentra más vinos buenos, pero acierta menos entre los que marca. La línea gris "
                f"({100 * y.mean():.0f} %) es lo que lograría alguien que marcara vinos al azar.")
    curvas = tabla_app("calibracion_curvas.csv")
    cal = tabla_app("calibracion.csv")
    if modelo != "SVM" and curvas is not None and (curvas["modelo"] == modelo).any():
        fila_cal = cal.set_index("modelo").loc[modelo]
        fig_cal = plots.curva_calibracion(curvas[curvas["modelo"] == modelo], modelo)
        ece = float(fila_cal["ece"])
        titular = ("Sus probabilidades son de fiar." if ece < 0.02 else "Sus probabilidades se desvían un poco de la realidad." if ece < 0.05
                   else "Sus probabilidades no se pueden leer al pie de la letra.")
        l_cal = idea(titular, f"En promedio se desvían {100 * ece:.1f} puntos porcentuales de lo que de verdad pasa (", termino("ECE"),
                     f" = {ece:.3f}). Si dice 30 %, deberían ser buenos unos 30 de cada 100 vinos. ", termino("Brier"),
                     f" = {fila_cal['brier']:.3f}.")
    else:
        fig_cal = plots.vacia("SVM entrega una función de decisión, no una probabilidad")
        l_cal = idea("SVM no entrega probabilidades.", "Su salida es un puntaje sin escala, así que no tiene curva de calibración.")
    return (plots.curva_roc(fpr, tpr, auc, modelo), l_roc, plots.curva_pr(rec, prec, pr_auc, y.mean()), l_pr, fig_cal, l_cal, contenido_shap(modelo))


clientside_callback(ClientsideFunction("cata", "decision"),
                    Output("dec-vp", "children"), Output("dec-fn", "children"), Output("dec-fp", "children"), Output("dec-vn", "children"),
                    Output("dec-precision", "children"), Output("dec-sensibilidad", "children"), Output("dec-f1", "children"),
                    Output("dec-umbral-texto", "children"), Output("dec-sugerencia", "children"), Output("dec-detalle", "children"),
                    Input("dec-datos", "data"), Input("diag-modelo", "value"), Input("dec-umbral", "value"), Input("dec-costo", "value"),
                    Input("g-roc", "figure"))


@callback(Output("duelo", "children"), Output("jueces", "children"), Input("duelo-rival", "value"))
def comparar(rival):
    mejor = json_app("veredicto.json")["mejor_benchmark"]
    return duelo(rival), jueces(rival, mejor)


clientside_callback(ClientsideFunction("cata", "nucleo"), Output("nucleo-texto", "children"),
                    Input("nucleo-ancho", "value"), Input("nucleo-delta", "value"), Input("nucleo-escala", "data"))
