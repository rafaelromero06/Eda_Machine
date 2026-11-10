import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos, json_app, tabla_app
from src.tablero import AUTORES, ORIGINAL, VERSION, cifra, encabezado, escala_notas, numero, recorrido, siguiente, tarjeta, termino

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Cata Ciega", order=0)

DF = cargar_vinos()
N_VINOS = len(DF)
BUENO = DF["quality"] >= 7
N_BUENOS = int(BUENO.sum())
PREVALENCIA = N_BUENOS / N_VINOS
PCT_NOTAS = {int(n): float(p) for n, p in (100 * DF["quality"].value_counts(normalize=True)).sort_index().items()}
OIV = "https://www.oiv.int/sites/default/files/documents/Annual_Assessment_2023.pdf"
UCI = "https://archive.ics.uci.edu/dataset/186/wine+quality"
SIMBOLOS = {"fixed acidity": ("Af", None), "volatile acidity": ("Av", None), "citric acid": ("Ci", None), "residual sugar": ("Az", None),
            "chlorides": ("Cl", None), "free sulfur dioxide": ("SO", "2"), "total sulfur dioxide": ("SO", "2"), "density": ("ρ", None),
            "pH": ("pH", None), "sulphates": ("Su", None), "alcohol": ("Al", None)}
RUTA = [
    ("Limpiar los datos", "Se unieron tintos y blancos y se quitaron 1177 vinos repetidos: quedan 5320.", "ETL", "Entregable 1"),
    ("Explorarlos", "Cómo se reparte cada medición y qué distingue a los vinos buenos, que son solo el 19 %.", "EDA", "Entregable 1"),
    ("Entrenar modelos", "7 tipos de modelo y 4 formas de compensar que hay pocos vinos buenos: 136 combinaciones.", "Modelos y balanceo",
     "Entregable 2"),
    ("Ajustarlos", "Cada modelo busca su mejor configuración con 4 métodos y el mismo número de intentos: 40.", "Optimización", "Entregable 2"),
    ("Examinarlos sin trampa", "Cada modelo se evalúa 25 veces con vinos que no usó para aprender.", "Validación anidada", "Entregables 2 y 3"),
    ("Separar mérito de suerte", "Pruebas que dicen si la ventaja de un modelo es real o pudo ser casualidad.", "Comparación estadística",
     "Entregables 2 y 3"),
    ("Proponer uno propio", "El Bosque Ordinal Focalizado: programado desde cero y con reglas fijadas antes de ver los resultados.",
     "Modelo original", "Entregable 3"),
    ("Ponerlo a prueba", "Otros 4 conjuntos de datos, sus piezas quitadas una a una y pruebas con ruido.", "Evaluación ampliada", "Entregable 3"),
    ("Publicarlo", "Los modelos guardados, este tablero en la web y el informe completo.", "Despliegue", "Entregable 3"),
]


def tendencia(dif):
    """Frase y clase de la flecha de una medición según cuánto difiere entre vinos buenos y no buenos."""
    if abs(dif) < 0.1:
        return "≈ casi igual en los buenos", "igual"
    return ("▲ más alto en los buenos", "sube") if dif > 0 else ("▼ más bajo en los buenos", "baja")


def elemento(c, puesto, r, dif):
    simbolo, sub = SIMBOLOS[c]
    nombre, unidad, descripcion = DICCIONARIO_VINO[c]
    frase, clase = tendencia(dif)
    m_b, m_n = DF.loc[BUENO, c].mean(), DF.loc[~BUENO, c].mean()
    return html.Div(html.Div([
        html.Div([html.Div(f"n.º {puesto}", className="elemento-numero"),
                  html.Div([simbolo, html.Sub(sub)] if sub else simbolo, className="elemento-simbolo"),
                  html.Div(nombre, className="elemento-nombre"), html.Div(frase, className=f"elemento-tendencia {clase}"),
                  html.Div(unidad, className="elemento-unidad")], className="cara"),
        html.Div([html.Div(descripcion),
                  html.Div([f"Buenos: {m_b:.3g}", html.Br(), f"Demás: {m_n:.3g}", html.Br(), f"r = {r:+.2f} con la nota"], className="correlacion-mini")],
                 className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0,
        **{"aria-label": f"{nombre} ({unidad}). {descripcion} Promedio en los vinos buenos: {m_b:.3g}; en los demás: {m_n:.3g}. "
                         f"Correlación con la nota: {r:+.2f}."})


def tabla_periodica():
    r = DF[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality")
    dif = (DF.loc[BUENO, NUMERICAS_VINO].mean() - DF.loc[~BUENO, NUMERICAS_VINO].mean()) / DF[NUMERICAS_VINO].std()
    orden = r.abs().sort_values(ascending=False).index
    tintos = DF["tipo_vino"] == "red"
    piezas = [elemento(c, i + 1, r[c], dif[c]) for i, c in enumerate(orden)]
    piezas.append(html.Div(html.Div([
        html.Div([html.Div("n.º 12", className="elemento-numero"), html.Div("Tv", className="elemento-simbolo"),
                  html.Div("Tipo de vino", className="elemento-nombre"), html.Div("más buenos entre los blancos", className="elemento-tendencia igual"),
                  html.Div("tinto / blanco", className="elemento-unidad")], className="cara"),
        html.Div([html.Div(f"{numero(int(tintos.sum()))} tintos y {numero(int((~tintos).sum()))} blancos."),
                  html.Div([f"Buenos entre los tintos: {100 * BUENO[tintos].mean():.0f} %", html.Br(),
                            f"entre los blancos: {100 * BUENO[~tintos].mean():.0f} %"], className="correlacion-mini")], className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0, **{"aria-label": "Tipo de vino: tinto o blanco."}))
    piezas.append(html.Div(html.Div([
        html.Div([html.Div("objetivo", className="elemento-numero"), html.Div("Q", className="elemento-simbolo"),
                  html.Div("Nota de calidad", className="elemento-nombre"), html.Div("lo que se quiere anticipar", className="elemento-tendencia igual"),
                  html.Div("0 a 10 · bueno si ≥ 7", className="elemento-unidad")],
                 className="cara objetivo"),
        html.Div([html.Div("Mediana de al menos tres catadores. Es la respuesta que los modelos intentan anticipar."),
                  html.Div(f"{numero(N_BUENOS)} vinos buenos de {numero(N_VINOS)}.", className="correlacion-mini")], className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0, **{"aria-label": "Nota de calidad, la variable objetivo."}))
    return html.Div(piezas, className="periodica")


def estacion(i, titulo, texto, tecnico, entrega):
    return html.Div([html.Div(str(i), className="estacion-num"), html.Div(titulo, className="estacion-titulo"),
                     html.Div(texto, className="estacion-texto"), html.Div(tecnico, className="estacion-tecnico"),
                     html.Div(entrega, className="entrega")], className="estacion aparecer")


def conclusion():
    v = json_app("veredicto.json")
    res = tabla_app("lideres.csv")
    if v is None or res is None:
        return html.P("El resumen de resultados se publica al terminar el análisis.")
    rechazos = sum(v["pruebas_que_rechazan"].values())
    puesto = int(res.sort_values("auc_media", ascending=False)["modelo"].tolist().index(ORIGINAL)) + 1
    parejas = round(100 * v["auc_bof"])
    cifras = [cifra(parejas, ["parejas de vinos (uno bueno y uno que no) bien ordenadas por el modelo propio. Es lo que mide el ", termino("AUC"),
                              f": {v['auc_bof']:.3f}."], sufijo=" de 100", i=0),
              cifra(puesto, "en desempeño promedio entre los modelos comparados.", sufijo=".º de 10", i=1),
              cifra(rechazos, "pruebas estadísticas confirman que su ventaja es real. Se exigían 3.", sufijo=" de 4", i=2),
              cifra(v["conjuntos_donde_bof_gana"], "conjuntos de datos distintos del vino en los que también gana.", sufijo=" de 4", i=3)]
    if v["superior"]:
        titular = "El modelo propio es superior a los 9 de referencia, según la regla fijada antes de los experimentos."
    else:
        titular = "El modelo propio compite con los mejores, pero no los supera con la certeza que se exigió de antemano."
    texto = (f"En vino quedó en el puesto {puesto} de 10, con un AUC de {v['auc_bof']:.3f} frente a {v['auc_mejor_benchmark']:.3f} de "
             f"{v['mejor_benchmark']}: una diferencia tan pequeña que solo {rechazos} de las 4 pruebas la distinguen del azar, y en los otros "
             f"4 conjuntos de datos ganó en {v['conjuntos_donde_bof_gana']}. Decirlo así, sin adornos, también es un resultado.")
    return html.Div([html.Div(cifras, className="cifras cifras-anchas mb-4"),
                     html.P(titular, className="conclusion-titular"), html.P(texto, className="mb-3", style={"maxWidth": "50rem"}),
                     dcc.Link([html.I(className="bi bi-patch-check"), "Ver el veredicto completo"], href="/modelos?pestana=original",
                              className="boton")])


layout = html.Div([
    html.Section(dbc.Container(dbc.Row([
        dbc.Col([
            html.Div("Vinho Verde · Portugal · 2004–2007", className="antetitulo"),
            html.H1(["Un sommelier que ", html.Em("nunca ha probado"), " un vino"]),
            html.P("A partir de 11 mediciones de laboratorio, los modelos de este proyecto anticipan si un vino recibirá una nota de 7 o "
                   "más en la cata. Aquí puedes entender cómo lo hacen, probarlos con tus propios vinos y decidir cuánto confiar en ellos.",
                   className="portada-lead"),
            html.Div([dcc.Link([html.I(className="bi bi-magic"), "Predecir un vino"], href="/modelos?pestana=prediccion", className="boton boton-claro"),
                      dcc.Link([html.I(className="bi bi-controller"), "Jugar a la cata a ciegas"], href="/cata", className="boton boton-contorno")],
                     className="d-flex flex-wrap gap-2 mt-4 mb-4"),
            html.Div([cifra(N_VINOS, "vinos analizados", i=0), cifra(11, "mediciones de laboratorio por vino", i=1),
                      cifra(10, "modelos comparados: 9 conocidos y 1 propio", i=2), cifra(25, "exámenes por modelo, con vinos que no vio", i=3)],
                     className="cifras"),
        ], lg=7),
        dbc.Col(html.Div([
            html.Canvas(className="copa", role="img", **{"data-nivel": f"{PREVALENCIA:.3f}", "data-tipo": "white", "data-tono": "claro",
                                                         "aria-label": f"Copa llena al {100 * PREVALENCIA:.0f} %: la proporción de vinos buenos"}),
            html.Div([html.Strong(f"{100 * PREVALENCIA:.0f} %"), "de los vinos son buenos: la copa está llena hasta ahí."], className="portada-nota"),
        ], className="portada-copa"), lg=5, class_name="mt-4 mt-lg-0"),
    ], class_name="align-items-center"), fluid="xl"), className="portada"),

    html.Section(dbc.Container([
        html.Div([html.Span("¿Por dónde empiezo?", className="etiqueta-seccion mb-0"),
                  html.Span("El tablero se recorre en cinco pasos; puedes seguirlos en orden o saltar al que quieras.", className="inicio-texto")],
                 className="inicio-cabecera"),
        recorrido("/"),
    ], fluid="xl"), className="banda-inicio"),

    html.Section(dbc.Container(dbc.Row([
        dbc.Col(encabezado("¿Por qué importa?", etiqueta="El problema"), lg=4),
        dbc.Col(html.Div([
            html.P(["En 2023 se produjeron 237 millones de hectolitros de vino en el mundo, la cifra más baja desde 1961 y un 10 % menos "
                    "que en 2022 ", html.A("(OIV, 2024)", href=OIV, target="_blank", rel="noopener"),
                    ". Con menos vino, cada lote cuenta y certificar su calidad a tiempo vale dinero."], className="fs-5"),
            html.P(["La calidad se certifica con paneles de catadores: es un proceso lento, costoso y subjetivo. Las pruebas de laboratorio, en "
                    "cambio, son rápidas y baratas. ", html.A("Cortez y colaboradores (2009)", href=UCI, target="_blank", rel="noopener"),
                    " mostraron que la química permite anticipar la nota. Este proyecto retoma esa idea con un objetivo práctico: ",
                    html.Strong("encontrar los vinos buenos antes de la cata"), "."]),
        ], className="aparecer"), lg=8),
    ]), fluid="xl"), className="banda banda-esmalte"),

    html.Section(dbc.Container([
        encabezado("¿Qué queremos responder?", etiqueta="Pregunta de investigación"),
        dbc.Row([
            dbc.Col([
                html.P(["¿Se puede saber si un vino es bueno solo con su química? ¿Y ", html.Em("ayuda"),
                        " usar la nota completa en lugar de reducirla a «bueno» o «no bueno»?"], className="pregunta aparecer"),
                html.P(["Los catadores ponen una nota de 0 a 10; en estos datos va de 3 a 9. Llamamos ", termino("bueno"),
                        " a un vino con 7 o más. La nota es ", termino("ordinal"), ": un 6 se quedó a un paso de ser bueno y un 3 está muy "
                        "lejos. Casi todos los modelos solo ven las dos cajas y pierden ese matiz."], className="aparecer"),
            ], lg=6, class_name="mb-4"),
            dbc.Col(html.Div([html.Div("Así se reparten los 5320 vinos por nota", className="escala-titulo"), escala_notas(PCT_NOTAS)],
                             className="tarjeta aparecer"), lg=6, class_name="mb-4"),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Comparar con justicia", "Los 7 modelos vistos en clase y 2 más recientes (LightGBM y CatBoost) presentan los mismos "
                                                     "exámenes, con el mismo número de intentos para ajustarse."), md=4, class_name="mb-3"),
            dbc.Col(tarjeta("Proponer un modelo propio", "Un bosque de árboles que aprovecha la nota completa, no solo el corte en 7, con las "
                                                         "reglas de evaluación fijadas antes de ver los resultados."), md=4, class_name="mb-3"),
            dbc.Col(tarjeta("Hacerlo útil", "Predicción en vivo con su explicación y su margen de duda, pensada para quien no es experto."),
                    md=4, class_name="mb-3"),
        ]),
    ], fluid="xl"), className="banda"),

    html.Section(dbc.Container([
        encabezado("¿De dónde salen los datos?", "Cada ficha es una medición de laboratorio. La flecha dice si los vinos buenos suelen tener más o "
                                                  "menos de eso, y el número, qué tan ligada está a la nota (1 es la más ligada). Pasa el cursor o "
                                                  "el foco del teclado para ver qué mide.", etiqueta="Los datos"),
        html.Div([
            html.Div([html.Div("Fuente", className="cifra-etiqueta"), html.A("UCI Machine Learning Repository", href=UCI, target="_blank", rel="noopener")], className="cifra"),
            html.Div([html.Div("Periodo y región", className="cifra-etiqueta"), html.Div("2004–2007, Vinho Verde (Portugal)")], className="cifra"),
            html.Div([html.Div("Licencia", className="cifra-etiqueta"), html.Div("CC BY 4.0")], className="cifra"),
            html.Div([html.Div("Observaciones", className="cifra-etiqueta"), html.Div("6497 vinos; 5320 sin repetidos")], className="cifra"),
            html.Div([html.Div("Variables", className="cifra-etiqueta"), html.Div("11 químicas y el tipo de vino")], className="cifra"),
            html.Div([html.Div("Tarea", className="cifra-etiqueta"), html.Div("Clasificación binaria: bueno si la nota es ≥ 7")], className="cifra"),
        ], className="cifras mb-4 aparecer"),
        tabla_periodica(),
    ], fluid="xl"), className="banda banda-esmalte"),

    html.Section(dbc.Container([
        encabezado("¿Cómo se hizo?", "Nueve pasos en tres entregas. En ninguno se usa información de los vinos con que después se examina a los "
                                     "modelos: así el resultado no sale inflado.", etiqueta="Metodología", oscuro=True),
        html.Div([html.Div(className="ruta-linea d-none d-xl-block", **{"aria-hidden": "true"})]
                 + [estacion(i + 1, *paso) for i, paso in enumerate(RUTA)], className="ruta"),
    ], fluid="xl"), className="banda banda-oscura con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.08"}),

    html.Section(dbc.Container([
        encabezado("¿A qué conclusión se llegó?", "El resumen de las tres entregas, sin adornos.", etiqueta="La respuesta corta"),
        html.Div(conclusion(), className="aparecer"),
    ], fluid="xl"), className="banda"),

    html.Section(dbc.Container(dbc.Row([
        dbc.Col(tarjeta("¿Qué límites tiene?", html.Ul([
            html.Li("La nota es la mediana de al menos tres catadores: es subjetiva y los modelos heredan ese ruido."),
            html.Li("Solo hay vinos de una región y de 2004 a 2007; no se sabe si las relaciones valen para otras regiones, uvas o años."),
            html.Li("Hay casi tres blancos por cada tinto y 1 de cada 5 vinos es bueno: los errores no se reparten igual entre grupos."),
            html.Li("No hay variables de uva, precio, marca ni añada, que también influyen en la calidad percibida."),
        ], className="mb-0")), lg=4, class_name="mb-3"),
        dbc.Col(tarjeta("¿Cómo usarlo con responsabilidad?", html.Ul([
            html.Li("El modelo apoya la evaluación sensorial; no la reemplaza ni certifica calidad."),
            html.Li("Una predicción no es una receta: cambiar la química de un vino no garantiza una mejor nota."),
            html.Li("Antes de usarlo en otra bodega o región hay que reentrenarlo y validarlo con datos locales."),
            html.Li(["Los conjuntos de datos de salud y educación de la evaluación ampliada solo sirvieron para comparar algoritmos; ",
                     "no para decidir sobre personas."]),
        ], className="mb-0")), lg=4, class_name="mb-3"),
        dbc.Col(tarjeta("Autores, versión y referencias", [
            html.P([html.Strong("Autores: "), f"{AUTORES} (Universidad del Norte)."]),
            html.P([html.Strong("Versión: "), VERSION, "."]),
            html.Ul([
                html.Li("Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). Modeling wine preferences by data mining from "
                        "physicochemical properties. Decision Support Systems, 47(4), 547–553."),
                html.Li(["OIV (2024). ", html.A("Annual Assessment of the World Vine and Wine Sector in 2023", href=OIV, target="_blank", rel="noopener"), "."]),
            ], className="small mb-0"),
        ]), lg=4, class_name="mb-3"),
    ]), fluid="xl"), className="banda banda-esmalte"),

    siguiente("/", "Ya conoces el problema. El siguiente paso es mirar los datos: qué distingue a un vino bueno."),
])
