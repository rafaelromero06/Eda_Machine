import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos
from src.original_model import auc_mann_whitney
from src.tablero import cifra, encabezado, lectura, numero, tarjeta, termino

dash.register_page(__name__, path="/eda", name="EDA", title="Exploración de los datos · Cata Ciega", order=1)

DF = cargar_vinos()
ETIQUETAS = {c: f"{DICCIONARIO_VINO[c][0]} ({DICCIONARIO_VINO[c][1]})" for c in NUMERICAS_VINO}
ETIQUETAS.update({"quality": "Nota de calidad", "tipo_vino": "Tipo de vino", "clase": "Clase (bueno / no bueno)"})
CORTAS = {c: DICCIONARIO_VINO[c][0] for c in NUMERICAS_VINO} | {"quality": "Nota"}
OPCIONES_VAR = [{"label": ETIQUETAS[c], "value": c} for c in NUMERICAS_VINO]
A_MIN, A_MAX = float(np.floor(DF["alcohol"].min())), float(np.ceil(DF["alcohol"].max()))
FILTROS = [Input("f-tipo", "value"), Input("f-nota", "value"), Input("f-alcohol", "value")]
DUPLICADOS = 1177


def filtrar(tipos, notas, alcohol):
    m = DF["tipo_vino"].isin(tipos or ["red", "white"]) & DF["quality"].between(*notas) & DF["alcohol"].between(*alcohol)
    return DF[m]


mesa = html.Div(dbc.Row([
    dbc.Col([html.Label("Tipo de vino", htmlFor="f-tipo", className="campo-etiqueta"),
             dcc.Dropdown(id="f-tipo", options=[{"label": "Tinto", "value": "red"}, {"label": "Blanco", "value": "white"}],
                          value=["red", "white"], multi=True, clearable=False, searchable=False)], md=4, class_name="mb-2"),
    dbc.Col([html.Label("Nota de calidad", htmlFor="f-nota", className="campo-etiqueta"),
             dcc.RangeSlider(id="f-nota", min=3, max=9, step=1, value=[3, 9], marks={i: str(i) for i in range(3, 10)})], md=4, class_name="mb-2"),
    dbc.Col([html.Label("Alcohol (% vol.)", htmlFor="f-alcohol", className="campo-etiqueta"),
             dcc.RangeSlider(id="f-alcohol", min=A_MIN, max=A_MAX, step=0.1, value=[A_MIN, A_MAX],
                             marks={i: str(i) for i in range(int(A_MIN), int(A_MAX) + 1, 2)},
                             tooltip={"placement": "bottom", "always_visible": False})], md=4, class_name="mb-2"),
]), className="mesa", role="search", **{"aria-label": "Filtros que afectan todas las gráficas de la página"})

layout = html.Div([
    html.Section(dbc.Container([
        html.Div("Análisis exploratorio", className="antetitulo"),
        html.H1("¿Qué distingue a un vino bueno?"),
        html.P("Los 5320 vinos sin duplicados que usan los modelos. Los filtros de abajo cambian todas las gráficas de la página; "
               "cada gráfica trae una frase que dice qué se ve."),
    ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
    html.Section(dbc.Container([
        mesa,
        html.Div(id="k-cifras", className="cifras mb-4"),
        dbc.Row([
            dbc.Col(tarjeta(None, [plots.grafico("g-objetivo", 360), lectura("", id="l-objetivo")]), lg=6, class_name="mb-4"),
            dbc.Col(tarjeta(None, [plots.grafico("g-huella", 440), lectura("", id="l-huella")]), lg=6, class_name="mb-4"),
        ]),
        encabezado("¿Qué variables van juntas?", ["Cada baldosa es la correlación de Pearson entre dos variables: azul si suben juntas, ocre si "
                                                  "una baja cuando la otra sube."], etiqueta="Relaciones"),
        tarjeta(None, [plots.grafico("g-corr", 560), lectura("", id="l-corr")], clase="mb-4"),
        encabezado("¿Cómo se reparte cada variable?", "Elige una variable para comparar su distribución en vinos buenos y no buenos.",
                   etiqueta="Distribuciones"),
        tarjeta(None, [
            dbc.Row(dbc.Col([html.Label("Variable", htmlFor="f-variable", className="campo-etiqueta"),
                             dcc.Dropdown(id="f-variable", options=OPCIONES_VAR, value="alcohol", clearable=False)], md=5), class_name="mb-2"),
            dbc.Row([dbc.Col(plots.grafico("g-hist", 360), lg=7), dbc.Col(plots.grafico("g-caja", 360), lg=5)]),
            lectura("", id="l-variable"),
        ], clase="mb-4"),
        encabezado("¿Cómo se relacionan dos variables?", "Elige los ejes y el color. Abajo, la misma pareja animada nota por nota.",
                   etiqueta="Explorador bivariado"),
        tarjeta(None, [
            dbc.Row([
                dbc.Col([html.Label("Eje x", htmlFor="f-x", className="campo-etiqueta"),
                         dcc.Dropdown(id="f-x", options=OPCIONES_VAR, value="alcohol", clearable=False)], md=4, class_name="mb-2"),
                dbc.Col([html.Label("Eje y", htmlFor="f-y", className="campo-etiqueta"),
                         dcc.Dropdown(id="f-y", options=OPCIONES_VAR, value="volatile acidity", clearable=False)], md=4, class_name="mb-2"),
                dbc.Col([html.Label("Color", htmlFor="f-color", className="campo-etiqueta"),
                         dcc.Dropdown(id="f-color", options=[{"label": "Clase (bueno / no bueno)", "value": "clase"},
                                                             {"label": "Tipo de vino", "value": "tipo_vino"},
                                                             {"label": "Nota de calidad", "value": "quality"}] + OPCIONES_VAR,
                                      value="clase", clearable=False)], md=4, class_name="mb-2"),
            ]),
            plots.grafico("g-disp", 460),
            lectura("", id="l-disp"),
        ], clase="mb-4"),
        tarjeta(None, [plots.grafico("g-danza", 500), lectura("", id="l-danza")], clase="mb-4"),
        encabezado("¿Faltan datos o hay valores raros?", etiqueta="Calidad de los datos"),
        tarjeta(None, [html.Div(id="t-calidad-datos"), lectura("", id="l-calidad-datos")]),
    ], fluid="xl"), className="banda banda-compacta"),
])


def tabla_calidad(d):
    filas = []
    for c in NUMERICAS_VINO:
        q1, q3 = d[c].quantile([0.25, 0.75])
        rango = q3 - q1
        n_at = int(((d[c] < q1 - 1.5 * rango) | (d[c] > q3 + 1.5 * rango)).sum())
        filas.append((DICCIONARIO_VINO[c][0], int(d[c].isna().sum()), n_at, 100 * n_at / len(d)))
    filas.sort(key=lambda f: -f[2])
    tope = max(f[3] for f in filas) or 1
    cuerpo = [html.Tr([html.Td(a), html.Td(b), html.Td(c),
                       html.Td(html.Div([html.Div(html.Div(style={"width": f"{100 * e / tope:.0f}%"}), className="pista-barra"),
                                         html.Span(f"{e:.1f} %", className="pista-texto")], className="d-flex align-items-center gap-2"))])
              for a, b, c, e in filas]
    tabla = dbc.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Faltantes"), html.Th("Atípicos (1.5 RIC)"), html.Th("% atípicos")])),
                       html.Tbody(cuerpo)], size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")
    texto = (f"No hay valores faltantes. La variable con más atípicos es {filas[0][0].lower()} ({filas[0][3]:.1f} %). Se conservan porque son "
             "mediciones plausibles y los modelos de árboles son poco sensibles a ellos.")
    return tabla, texto


@callback(Output("k-cifras", "children"), Output("g-objetivo", "figure"), Output("l-objetivo", "children"),
          Output("g-huella", "figure"), Output("l-huella", "children"), Output("g-corr", "figure"), Output("l-corr", "children"),
          Output("t-calidad-datos", "children"), Output("l-calidad-datos", "children"), *FILTROS)
def actualizar_general(tipos, notas, alcohol):
    d = filtrar(tipos, notas, alcohol)
    clave = f"{len(d)}-{notas}-{alcohol}-{tipos}"
    if len(d) < 10:
        vacia = plots.vacia("Muy pocos vinos con estos filtros")
        cifras = [cifra(len(d), "vinos con los filtros actuales")]
        return cifras, vacia, "Amplía los filtros.", vacia, "", vacia, "", "", ""
    buenos = (d["quality"] >= 7).mean()
    cifras = html.Div([cifra(len(d), "vinos con los filtros actuales", i=0), cifra(100 * buenos, "son buenos (nota ≥ 7)", 1, " %", i=1),
                       cifra(DUPLICADOS, "duplicados exactos eliminados antes de modelar", i=2), cifra(0, "valores faltantes", i=3)],
                      className="cifras", key=clave, style={"display": "contents"})
    lect_obj = (f"Con estos filtros, {100 * buenos:.1f} % de los {numero(len(d))} vinos son buenos. La nota más frecuente es "
                f"{int(d['quality'].mode().iloc[0])} y las notas extremas (3, 4, 8 y 9) suman solo {100 * d['quality'].isin([3, 4, 8, 9]).mean():.1f} %: "
                "casi todo se concentra en 5, 6 y 7, justo alrededor del corte.")
    if d["quality"].nunique() < 2 or (d["quality"] >= 7).all() or (d["quality"] < 7).all():
        huella, lect_huella = plots.vacia("Los filtros dejan una sola clase"), "Para comparar, incluye notas por debajo y por encima de 7."
    else:
        huella = plots.huella_quimica(d, NUMERICAS_VINO, CORTAS)
        bueno = d["quality"] >= 7
        dif = ((d.loc[bueno, NUMERICAS_VINO].mean() - d.loc[~bueno, NUMERICAS_VINO].mean()) / d[NUMERICAS_VINO].std()).dropna()
        arriba, abajo = dif.idxmax(), dif.idxmin()
        lect_huella = (f"Lo que más separa a los vinos buenos: más {DICCIONARIO_VINO[arriba][0].lower()} ({dif[arriba]:+.2f} desviaciones) "
                       f"y menos {DICCIONARIO_VINO[abajo][0].lower()} ({dif[abajo]:+.2f}). Por encima de 0.5 desviaciones la diferencia ya se nota a simple vista.")
    corr = d[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality").dropna()
    top = corr.abs().sort_values(ascending=False).index[:3]
    lect_corr = ("Las variables más asociadas con la nota son " + ", ".join(f"{DICCIONARIO_VINO[c][0].lower()} (r = {corr[c]:+.2f})" for c in top)
                 + (". Ninguna pasa de 0.5 en valor absoluto: una sola variable no basta y los modelos deben combinar varias."
                    if corr.abs().max() < 0.5 else f". La asociación más fuerte llega a |r| = {corr.abs().max():.2f} con estos filtros."))
    tabla, lect_cal = tabla_calidad(d)
    return (cifras, plots.distribucion_objetivo(d), lect_obj, huella, lect_huella, plots.correlaciones(d, NUMERICAS_VINO + ["quality"], CORTAS),
            lect_corr, tabla, lect_cal)


@callback(Output("g-hist", "figure"), Output("g-caja", "figure"), Output("l-variable", "children"), *FILTROS, Input("f-variable", "value"))
def actualizar_variable(tipos, notas, alcohol, variable):
    d = filtrar(tipos, notas, alcohol)
    bueno = d["quality"] >= 7
    if bueno.sum() < 3 or (~bueno).sum() < 3:
        vacia = plots.vacia("Los filtros dejan una sola clase")
        return vacia, vacia, "Para comparar clases, los filtros deben incluir notas por debajo y por encima de 7."
    m_b, m_n = d.loc[bueno, variable].median(), d.loc[~bueno, variable].median()
    nombre = DICCIONARIO_VINO[variable][0].lower()
    auc = auc_mann_whitney(bueno.to_numpy(), d[variable].to_numpy())
    separa = max(auc, 1 - auc)
    juicio = "separa bien las clases" if separa >= 0.70 else "separa algo las clases" if separa >= 0.60 else "casi no separa las clases"
    lectura_ = [f"La mediana de {nombre} es {m_b:.3g} en los vinos buenos y {m_n:.3g} en los no buenos. Usada sola, la variable {juicio}: su ",
                termino("AUC"), f" es {separa:.2f} (0.5 sería azar). ",
                "Los valores altos se asocian con vinos buenos." if auc >= 0.5 else "Los valores bajos se asocian con vinos buenos."]
    return plots.histograma(d, variable, ETIQUETAS[variable]), plots.caja_atipicos(d, variable, ETIQUETAS[variable]), lectura_


@callback(Output("g-disp", "figure"), Output("l-disp", "children"), Output("g-danza", "figure"), Output("l-danza", "children"),
          *FILTROS, Input("f-x", "value"), Input("f-y", "value"), Input("f-color", "value"))
def actualizar_dispersion(tipos, notas, alcohol, x, y, color):
    d = filtrar(tipos, notas, alcohol)
    if len(d) < 10:
        vacia = plots.vacia("Muy pocos vinos con estos filtros")
        return vacia, "", vacia, ""
    r = d[[x, y]].corr().iloc[0, 1]
    fuerza = "fuerte" if abs(r) >= 0.6 else "moderada" if abs(r) >= 0.3 else "débil"
    lect = (f"La correlación entre {DICCIONARIO_VINO[x][0].lower()} y {DICCIONARIO_VINO[y][0].lower()} es {r:+.2f} ({fuerza}). "
            "Mira si los puntos azules (vinos buenos) ocupan una zona propia del plano: si se mezclan con los ocres, esas dos variables no bastan.")
    if x == y:
        return plots.dispersion(d, x, y, color, ETIQUETAS), lect, plots.vacia("Elige dos variables distintas"), ""
    medianas = d.groupby("quality")[[x, y]].median()
    lect_danza = (f"Pulsa «Reproducir»: la nube recorre las notas de {int(medianas.index.min())} a {int(medianas.index.max())}. La mediana pasa de "
                  f"{medianas[x].iloc[0]:.3g} a {medianas[x].iloc[-1]:.3g} en {DICCIONARIO_VINO[x][0].lower()} y de {medianas[y].iloc[0]:.3g} a "
                  f"{medianas[y].iloc[-1]:.3g} en {DICCIONARIO_VINO[y][0].lower()}. La línea oscura es el camino que sigue la calidad.")
    return plots.dispersion(d, x, y, color, ETIQUETAS), lect, plots.danza_notas(d, x, y, ETIQUETAS), lect_danza
