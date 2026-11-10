import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, dcc, html

from src import plots
from src.data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos
from src.original_model import auc_mann_whitney
from src.tablero import cifra, como_leer, encabezado, idea, lectura, minuscula, numero, siguiente, tabla, tarjeta, termino

dash.register_page(__name__, path="/eda", name="Datos", title="Exploración de los datos · Cata Ciega", order=1)

DF = cargar_vinos()
ETIQUETAS = {c: f"{DICCIONARIO_VINO[c][0]} ({DICCIONARIO_VINO[c][1]})" for c in NUMERICAS_VINO}
ETIQUETAS.update({"quality": "Nota de calidad", "tipo_vino": "Tipo de vino", "clase": "Clase (bueno / no bueno)"})
CORTAS = {c: DICCIONARIO_VINO[c][0] for c in NUMERICAS_VINO} | {"quality": "Nota"}
OPCIONES_VAR = [{"label": ETIQUETAS[c], "value": c} for c in NUMERICAS_VINO]
A_MIN, A_MAX = float(np.floor(DF["alcohol"].min())), float(np.ceil(DF["alcohol"].max()))
FILTROS = [Input("f-tipo", "value"), Input("f-nota", "value"), Input("f-alcohol", "value")]
DUPLICADOS = 1177


def nombre(c):
    """Nombre de una variable listo para ir dentro de una frase."""
    return minuscula(DICCIONARIO_VINO[c][0])


def filtrar(tipos, notas, alcohol):
    m = DF["tipo_vino"].isin(tipos or ["red", "white"]) & DF["quality"].between(*notas) & DF["alcohol"].between(*alcohol)
    return DF[m]


def diferencias(d):
    """Diferencia entre vinos buenos y no buenos en cada medición, en desviaciones estándar."""
    bueno = d["quality"] >= 7
    return ((d.loc[bueno, NUMERICAS_VINO].mean() - d.loc[~bueno, NUMERICAS_VINO].mean()) / d[NUMERICAS_VINO].std()).dropna()


def resumen():
    """Las tres ideas que deja la exploración, calculadas con los 5320 vinos."""
    bueno = DF["quality"] >= 7
    dif = diferencias(DF)
    c = dif.abs().idxmax()
    unidad = DICCIONARIO_VINO[c][1]
    r = DF[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality").abs()
    ideas = [
        ("Los vinos buenos son minoría", f"Solo {100 * bueno.mean():.0f} de cada 100 vinos tienen nota de 7 o más. Un modelo que dijera siempre "
                                         "«no bueno» acertaría casi siempre y no serviría de nada: por eso no se mide con el porcentaje de aciertos."),
        (f"El {nombre(c)} es lo que más los distingue", f"Los vinos buenos promedian {DF.loc[bueno, c].mean():.3g} {unidad} y los demás "
                                                        f"{DF.loc[~bueno, c].mean():.3g}. Ninguna otra medición los separa tanto."),
        ("Ninguna medición basta sola", f"La más ligada a la nota apenas llega a una correlación de {r.max():.2f}. Hay que combinar varias a la "
                                        "vez, y eso es justo lo que hacen los modelos."),
    ]
    return dbc.Row([dbc.Col(html.Div([html.Div(str(i + 1), className="idea-num"), html.H3(t, className="tarjeta-titulo"), html.P(x, className="mb-0")],
                                     className="tarjeta idea-clave aparecer"), lg=4, class_name="mb-3") for i, (t, x) in enumerate(ideas)])


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
        html.Div("Análisis exploratorio de datos (EDA)", className="antetitulo"),
        html.H1("¿Qué distingue a un vino bueno?"),
        html.P("Estos son los 5320 vinos con los que aprenden los modelos. Mueve los filtros: todas las gráficas cambian con ellos y cada una "
               "trae su conclusión en una frase."),
    ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
    html.Section(dbc.Container([
        mesa,
        html.Div(id="k-cifras", className="cifras mb-4"),
        dbc.Row([
            dbc.Col(tarjeta(None, [plots.grafico("g-objetivo", 360), lectura("", id="l-objetivo")]), lg=6, class_name="mb-4"),
            dbc.Col(tarjeta(None, [plots.grafico("g-huella", 440), lectura("", id="l-huella")]), lg=6, class_name="mb-4"),
        ]),
        encabezado("¿Qué mediciones van juntas?", ["Cada baldosa compara dos mediciones con su ", termino("correlación"),
                                                   ": azul si suben juntas, ocre si una baja cuando la otra sube y clara si no se relacionan."],
                   etiqueta="Relaciones"),
        tarjeta(None, [plots.grafico("g-corr", 560),
                       como_leer("empieza por la fila enmarcada, «Nota»: dice qué tan ligada está cada medición a la calidad. El resto muestra "
                                 "qué mediciones cuentan casi lo mismo, como la densidad y el alcohol."),
                       lectura("", id="l-corr")], clase="mb-4"),
        encabezado("¿Cómo se reparte cada medición?", "Elige una medición y compara cómo se reparte en los vinos buenos y en los demás.",
                   etiqueta="Distribuciones"),
        tarjeta(None, [
            dbc.Row(dbc.Col([html.Label("Medición", htmlFor="f-variable", className="campo-etiqueta"),
                             dcc.Dropdown(id="f-variable", options=OPCIONES_VAR, value="alcohol", clearable=False)], md=5), class_name="mb-2"),
            dbc.Row([dbc.Col(plots.grafico("g-hist", 360), lg=7), dbc.Col(plots.grafico("g-caja", 360), lg=5)]),
            como_leer("a la izquierda, cuanto menos se pisan el color azul y el ocre, mejor separa esa medición. A la derecha, cada caja reúne "
                      "a la mitad central de los vinos y los puntos sueltos son valores poco comunes."),
            lectura("", id="l-variable"),
        ], clase="mb-4"),
        encabezado("¿Cómo se relacionan dos mediciones?", "Elige qué va en cada eje y qué pinta el color. Abajo, la misma pareja se anima nota por nota.",
                   etiqueta="Explorador bivariado"),
        tarjeta(None, [
            dbc.Row([
                dbc.Col([html.Label("Eje horizontal", htmlFor="f-x", className="campo-etiqueta"),
                         dcc.Dropdown(id="f-x", options=OPCIONES_VAR, value="alcohol", clearable=False)], md=4, class_name="mb-2"),
                dbc.Col([html.Label("Eje vertical", htmlFor="f-y", className="campo-etiqueta"),
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
    html.Section(dbc.Container([
        encabezado("¿Qué nos llevamos de los datos?", "Tres ideas, con los 5320 vinos, que explican las decisiones del modelado.",
                   etiqueta="En resumen"),
        resumen(),
    ], fluid="xl"), className="banda banda-esmalte"),
    siguiente("/eda", "Ya sabes qué distingue a un vino bueno. Ahora, los modelos: cuál acierta más y cómo predicen un vino."),
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
    cuerpo = [(a, b, c, html.Div([html.Div(html.Div(style={"width": f"{100 * e / tope:.0f}%"}), className="pista-barra"),
                                  html.Span(f"{e:.1f} %", className="pista-texto")], className="d-flex align-items-center gap-2"))
              for a, b, c, e in filas]
    columnas = [("Medición", None), ("Datos faltantes", "Vinos sin valor en esa medición"),
                ("Valores poco comunes", "Atípicos: a más de 1.5 veces el rango intercuartílico (RIC) de la caja"),
                ("% de los vinos", "Porcentaje de vinos con un valor poco común en esa medición")]
    faltantes = sum(f[1] for f in filas)
    primera = "No falta ningún dato." if faltantes == 0 else f"Faltan {faltantes} datos."
    texto = idea(primera, f"La medición con más valores poco comunes es {minuscula(filas[0][0])} ({filas[0][3]:.1f} % de los vinos). Se conservan "
                          "porque son mediciones posibles, no errores, y los modelos de árboles casi no se alteran con un valor ",
                 termino("atípico"), ".")
    return tabla(columnas, cuerpo), texto


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
    cifras = html.Div([cifra(len(d), "vinos con los filtros actuales", i=0), cifra(100 * buenos, "son buenos (nota de 7 o más)", 1, " %", i=1),
                       cifra(DUPLICADOS, "vinos repetidos que se quitaron antes de modelar", i=2), cifra(0, "datos faltantes", i=3)],
                      className="cifras", key=clave, style={"display": "contents"})
    extremas = 100 * d["quality"].isin([3, 4, 8, 9]).mean()
    lect_obj = idea(f"De cada 100 vinos, {100 * buenos:.0f} son buenos.",
                    f"Con estos filtros hay {numero(len(d))} vinos y {100 * buenos:.1f} % tiene nota de 7 o más. La nota más frecuente es "
                    f"{int(d['quality'].mode().iloc[0])} y las extremas (3, 4, 8 y 9) suman solo {extremas:.1f} %: casi todo se concentra "
                    "alrededor del corte, donde un vino de 6 y uno de 7 se parecen mucho.")
    if d["quality"].nunique() < 2 or (d["quality"] >= 7).all() or (d["quality"] < 7).all():
        huella, lect_huella = plots.vacia("Los filtros dejan una sola clase"), "Para comparar, incluye notas por debajo y por encima de 7."
    else:
        huella = plots.huella_quimica(d, NUMERICAS_VINO, CORTAS)
        bueno = d["quality"] >= 7
        dif = diferencias(d)
        arriba, abajo = dif.idxmax(), dif.idxmin()
        medias = lambda c: f"{d.loc[bueno, c].mean():.3g} frente a {d.loc[~bueno, c].mean():.3g} {DICCIONARIO_VINO[c][1]}"
        if dif[arriba] > 0 > dif[abajo]:
            titular = f"Los vinos buenos tienen más {nombre(arriba)} y menos {nombre(abajo)}."
            detalle = [f"En {nombre(arriba)} promedian {medias(arriba)}; en {nombre(abajo)}, {medias(abajo)}. "]
        else:
            mayor = dif.abs().idxmax()
            titular = f"La mayor diferencia entre vinos buenos y no buenos está en {nombre(mayor)}."
            detalle = [f"Promedian {medias(mayor)}. "]
        lect_huella = idea(titular, *detalle, "Cada punto mide la diferencia en ", termino("desviaciones estándar", "desviación estándar"),
                           ", para poder comparar mediciones con unidades distintas: por encima de 0.5 ya se nota a simple vista.")
    corr = d[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality").dropna()
    top = corr.abs().sort_values(ascending=False).index[:3]
    lista = ", ".join(f"{nombre(c)} (r = {corr[c]:+.2f})" for c in top)
    if corr.abs().max() < 0.5:
        lect_corr = idea("Ninguna medición basta por sí sola para saber la nota.",
                         f"Las más ligadas a la nota son {lista}. Ninguna pasa de 0.5: hay que combinar varias, y eso hacen los modelos.")
    else:
        lect_corr = idea(f"Con estos filtros, {nombre(top[0])} está claramente ligado a la nota.", f"Las más ligadas son {lista}.")
    tabla_datos, lect_cal = tabla_calidad(d)
    return (cifras, plots.distribucion_objetivo(d), lect_obj, huella, lect_huella, plots.correlaciones(d, NUMERICAS_VINO + ["quality"], CORTAS),
            lect_corr, tabla_datos, lect_cal)


@callback(Output("g-hist", "figure"), Output("g-caja", "figure"), Output("l-variable", "children"), *FILTROS, Input("f-variable", "value"))
def actualizar_variable(tipos, notas, alcohol, variable):
    d = filtrar(tipos, notas, alcohol)
    bueno = d["quality"] >= 7
    if bueno.sum() < 3 or (~bueno).sum() < 3:
        vacia = plots.vacia("Los filtros dejan una sola clase")
        return vacia, vacia, "Para comparar clases, los filtros deben incluir notas por debajo y por encima de 7."
    m_b, m_n = d.loc[bueno, variable].median(), d.loc[~bueno, variable].median()
    nom = nombre(variable)
    auc = auc_mann_whitney(bueno.to_numpy(), d[variable].to_numpy())
    separa = max(auc, 1 - auc)
    if separa >= 0.70:
        titular = f"Con solo mirar {nom} ya se distingue bastante bien a los vinos buenos."
    elif separa >= 0.60:
        titular = f"Mirar {nom} ayuda algo a distinguir los vinos buenos, pero no alcanza."
    else:
        titular = f"Mirar {nom} casi no sirve para distinguir los vinos buenos."
    sentido = "Los valores altos se asocian con vinos buenos." if auc >= 0.5 else "Los valores bajos se asocian con vinos buenos."
    lectura_ = idea(titular, "El valor típico (la ", termino("mediana"), f") es {m_b:.3g} en los buenos y {m_n:.3g} en los demás. {sentido} Usando "
                    f"solo esta medición se ordenarían bien {100 * separa:.0f} de cada 100 parejas de vinos (", termino("AUC"),
                    f" = {separa:.2f}; 50 de 100 sería adivinar).")
    return plots.histograma(d, variable, ETIQUETAS[variable]), plots.caja_atipicos(d, variable, ETIQUETAS[variable]), lectura_


@callback(Output("g-disp", "figure"), Output("l-disp", "children"), Output("g-danza", "figure"), Output("l-danza", "children"),
          *FILTROS, Input("f-x", "value"), Input("f-y", "value"), Input("f-color", "value"))
def actualizar_dispersion(tipos, notas, alcohol, x, y, color):
    d = filtrar(tipos, notas, alcohol)
    if len(d) < 10:
        vacia = plots.vacia("Muy pocos vinos con estos filtros")
        return vacia, "", vacia, ""
    r = d[[x, y]].corr().iloc[0, 1]
    if abs(r) >= 0.6:
        titular = f"{DICCIONARIO_VINO[x][0]} y {nombre(y)} van muy ligados."
    elif abs(r) >= 0.3:
        titular = f"{DICCIONARIO_VINO[x][0]} y {nombre(y)} tienen una relación moderada."
    else:
        titular = f"{DICCIONARIO_VINO[x][0]} y {nombre(y)} casi no se relacionan entre sí."
    lect = idea(titular, f"Su correlación es {r:+.2f}. Fíjate si los puntos azules (vinos buenos) ocupan una zona propia: si se mezclan con los "
                         "ocres, esas dos mediciones no bastan para separarlos.")
    if x == y:
        return plots.dispersion(d, x, y, color, ETIQUETAS), lect, plots.vacia("Elige dos mediciones distintas"), ""
    medianas = d.groupby("quality")[[x, y]].median()
    n0, n1 = int(medianas.index.min()), int(medianas.index.max())
    lect_danza = idea(f"De la nota {n0} a la {n1}, el valor típico de {nombre(x)} pasa de {medianas[x].iloc[0]:.3g} a {medianas[x].iloc[-1]:.3g}.",
                      f"En {nombre(y)} pasa de {medianas[y].iloc[0]:.3g} a {medianas[y].iloc[-1]:.3g}. Pulsa «Reproducir» para ver la nube de "
                      "vinos de cada nota; la línea oscura es el camino que sigue el vino típico al subir la calidad.")
    return plots.dispersion(d, x, y, color, ETIQUETAS), lect, plots.danza_notas(d, x, y, ETIQUETAS), lect_danza
