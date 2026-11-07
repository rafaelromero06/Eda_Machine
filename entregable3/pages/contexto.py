import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data import DICCIONARIO_VINO, NUMERICAS_VINO, cargar_vinos, json_app, tabla_app
from src.tablero import AUTORES, VERSION, cifra, encabezado, tarjeta

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Cata Ciega", order=0)

DF = cargar_vinos()
N_VINOS = len(DF)
N_BUENOS = int((DF["quality"] >= 7).sum())
PREVALENCIA = N_BUENOS / N_VINOS
ORIGINAL = "Bosque Ordinal Focalizado"
OIV = "https://www.oiv.int/sites/default/files/documents/Annual_Assessment_2023.pdf"
UCI = "https://archive.ics.uci.edu/dataset/186/wine+quality"
SIMBOLOS = {"fixed acidity": ("Af", None), "volatile acidity": ("Av", None), "citric acid": ("Ci", None), "residual sugar": ("Az", None),
            "chlorides": ("Cl", None), "free sulfur dioxide": ("SO", "2"), "total sulfur dioxide": ("SO", "2"), "density": ("ρ", None),
            "pH": ("pH", None), "sulphates": ("Su", None), "alcohol": ("Al", None)}
RUTA = [
    ("ETL", "Unión de tintos y blancos y eliminación de 1177 duplicados exactos: quedan 5320 vinos.", "Entregable 1"),
    ("EDA", "Distribuciones, correlaciones, atípicos y balance de clases: solo el 19 % de los vinos es bueno.", "Entregable 1"),
    ("Modelos", "7 algoritmos con 4 técnicas de balanceo y su versión de regresión: 136 combinaciones.", "Entregable 2"),
    ("Optimización", "Grid Search, Random Search, bayesiana y genética, con 40 evaluaciones por búsqueda.", "Entregable 2"),
    ("Validación anidada", "5 × 3 pliegues; en el Entregable 3 se repite 5 veces: 25 bloques pareados.", "Entregables 2 y 3"),
    ("Comparación estadística", "Friedman, Nemenyi, Wilcoxon, t corregida, DeLong y MCS, con ajuste de Holm.", "Entregables 2 y 3"),
    ("Modelo original", "Bosque Ordinal Focalizado: programado desde cero y evaluado con un protocolo pre-registrado.", "Entregable 3"),
    ("Evaluación ampliada", "4 conjuntos de datos adicionales, ablación y estudios de robustez.", "Entregable 3"),
    ("Despliegue", "Modelos serializados, este tablero en Render y el informe en Jupyter Book.", "Entregable 3"),
]


def elemento(c, numero, r):
    simbolo, sub = SIMBOLOS[c]
    nombre, unidad, descripcion = DICCIONARIO_VINO[c]
    signo = "más alto" if r > 0 else "más bajo"
    return html.Div(html.Div([
        html.Div([html.Div(f"n.º {numero}", className="elemento-numero"),
                  html.Div([simbolo, html.Sub(sub)] if sub else simbolo, className="elemento-simbolo"),
                  html.Div(nombre, className="elemento-nombre"), html.Div(unidad, className="elemento-unidad")], className="cara"),
        html.Div([html.Div(descripcion),
                  html.Div(f"r = {r:+.2f} con la nota: los vinos con nota alta tienden a tenerlo {signo}.", className="correlacion-mini")],
                 className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0,
        **{"aria-label": f"{nombre} ({unidad}). {descripcion} Correlación con la nota: {r:+.2f}."})


def tabla_periodica():
    r = DF[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality")
    orden = r.abs().sort_values(ascending=False).index
    tintos = int((DF["tipo_vino"] == "red").sum())
    piezas = [elemento(c, i + 1, r[c]) for i, c in enumerate(orden)]
    piezas.append(html.Div(html.Div([
        html.Div([html.Div("n.º 12", className="elemento-numero"), html.Div("Tv", className="elemento-simbolo"),
                  html.Div("Tipo de vino", className="elemento-nombre"), html.Div("tinto / blanco", className="elemento-unidad")], className="cara"),
        html.Div([html.Div(f"{tintos:,} tintos y {N_VINOS - tintos:,} blancos.".replace(",", ".")),
                  html.Div("Los blancos tienen más vinos buenos que los tintos.", className="correlacion-mini")], className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0, **{"aria-label": "Tipo de vino: tinto o blanco."}))
    piezas.append(html.Div(html.Div([
        html.Div([html.Div("objetivo", className="elemento-numero"), html.Div("Q", className="elemento-simbolo"),
                  html.Div("Nota de calidad", className="elemento-nombre"), html.Div("0 a 10 · bueno si ≥ 7", className="elemento-unidad")],
                 className="cara", style={"background": "#e4edf9"}),
        html.Div([html.Div("Mediana de al menos tres catadores. Es la respuesta que los modelos intentan anticipar."),
                  html.Div(f"{N_BUENOS:,} vinos buenos de {N_VINOS:,}.".replace(",", "."), className="correlacion-mini")], className="dorso"),
    ], className="elemento-interno"), className="elemento", tabIndex=0, **{"aria-label": "Nota de calidad, la variable objetivo."}))
    return html.Div(piezas, className="periodica")


def estacion(i, titulo, texto, entrega):
    return html.Div([html.Div(str(i), className="estacion-num"), html.Div(titulo, className="estacion-titulo"),
                     html.Div(texto, className="estacion-texto"), html.Div(entrega, className="entrega")], className="estacion aparecer")


def conclusion():
    v = json_app("veredicto.json")
    res = tabla_app("lideres.csv")
    if v is None or res is None:
        return html.P("El resumen de resultados se publica al terminar el análisis.")
    rechazos = sum(v["pruebas_que_rechazan"].values())
    rango = tabla_app("rangos_friedman.csv")
    puesto = int(res.sort_values("auc_media", ascending=False)["modelo"].tolist().index(ORIGINAL)) + 1
    texto = (f"En vino, el modelo original quedó en el puesto {puesto} de 10 por AUC medio ({v['auc_bof']:.3f} frente a "
             f"{v['auc_mejor_benchmark']:.3f} de {v['mejor_benchmark']}). La diferencia es pequeña: {rechazos} de las 4 pruebas "
             f"pre-registradas la confirman y en los conjuntos adicionales ganó en {v['conjuntos_donde_bof_gana']} de 4. ")
    texto += ("Según la regla fijada antes de los experimentos, es superior." if v["superior"] else
              "Según la regla fijada antes de los experimentos, no es superior: es competitivo, no mejor.")
    cifras = [cifra(v["auc_bof"], "AUC del modelo original en vino", 3, i=0),
              cifra(rechazos, "de 4 pruebas confirman la ventaja", i=1),
              cifra(v["conjuntos_donde_bof_gana"], "de 4 conjuntos adicionales ganados", i=2)]
    if rango is not None:
        r = rango.set_index(rango.columns[0])["rango_medio"]
        cifras.insert(1, cifra(float(r[ORIGINAL]), "rango medio de Friedman (1 = mejor)", 2, i=1))
    return html.Div([html.Div(cifras, className="cifras mb-3"), html.P(texto, className="mb-3", style={"maxWidth": "48rem"}),
                     dcc.Link([html.I(className="bi bi-patch-check me-2"), "Ver el veredicto completo"], href="/modelos?pestana=original",
                              className="boton boton-claro")])


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
            html.Div([cifra(N_VINOS, "vinos analizados", i=0), cifra(11, "mediciones químicas por vino", i=1),
                      cifra(136, "combinaciones en el torneo del Entregable 2", i=2), cifra(450, "validaciones del Entregable 3", i=3)],
                     className="cifras"),
        ], lg=7),
        dbc.Col(html.Div([
            html.Canvas(className="copa", role="img", **{"data-nivel": f"{PREVALENCIA:.3f}", "data-tipo": "white", "data-tono": "claro",
                                                         "aria-label": f"Copa llena al {100 * PREVALENCIA:.0f} %: la proporción de vinos buenos"}),
            html.Div([html.Strong(f"{100 * PREVALENCIA:.0f} %"), "de los vinos son buenos: la copa está llena hasta ahí."], className="portada-nota"),
        ], className="portada-copa"), lg=5, class_name="mt-4 mt-lg-0"),
    ], class_name="align-items-center"), fluid="xl"), className="portada"),

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
        html.P(["¿Qué tan bien se distingue un vino bueno (nota ≥ 7) de uno que no lo es usando solo su química, y ", html.Em("ayuda"),
                " aprovechar que la nota es ordinal en lugar de reducirla a bueno / no bueno?"],
               className="aparecer", style={"fontFamily": "var(--serif)", "fontSize": "clamp(1.4rem, 2.6vw, 2rem)", "lineHeight": "1.3",
                                            "maxWidth": "50rem"}),
        dbc.Row([
            dbc.Col(tarjeta("Comparar con rigor", "Los 7 modelos vistos en clase y 2 métodos recientes (LightGBM y CatBoost), con las mismas "
                                                  "particiones, el mismo presupuesto de búsqueda y pruebas estadísticas."), md=4, class_name="mb-3"),
            dbc.Col(tarjeta("Proponer un modelo propio", "Un bosque que usa toda la nota, no solo el corte en 7, evaluado con un protocolo "
                                                         "fijado antes de ver los resultados."), md=4, class_name="mb-3"),
            dbc.Col(tarjeta("Llevarlo a producción", "Predicción en vivo con incertidumbre y explicación de cada predicción, para usuarios "
                                                     "no expertos."), md=4, class_name="mb-3"),
        ], class_name="mt-2"),
    ], fluid="xl"), className="banda"),

    html.Section(dbc.Container([
        encabezado("¿De dónde salen los datos?", "Pasa el cursor (o el foco del teclado) sobre cada elemento para ver qué mide y cómo se "
                                                  "relaciona con la nota. El número indica su orden de asociación con la calidad.",
                   etiqueta="Los datos"),
        html.Div([
            html.Div([html.Div("Fuente", className="cifra-etiqueta"), html.A("UCI Machine Learning Repository", href=UCI, target="_blank", rel="noopener")], className="cifra"),
            html.Div([html.Div("Periodo y región", className="cifra-etiqueta"), html.Div("2004–2007, Vinho Verde (Portugal)")], className="cifra"),
            html.Div([html.Div("Licencia", className="cifra-etiqueta"), html.Div("CC BY 4.0")], className="cifra"),
            html.Div([html.Div("Observaciones", className="cifra-etiqueta"), html.Div("6497 vinos; 5320 sin duplicados exactos")], className="cifra"),
            html.Div([html.Div("Variables", className="cifra-etiqueta"), html.Div("11 químicas y el tipo de vino")], className="cifra"),
            html.Div([html.Div("Tarea", className="cifra-etiqueta"), html.Div("Clasificación binaria: bueno si la nota es ≥ 7")], className="cifra"),
        ], className="cifras mb-4 aparecer"),
        tabla_periodica(),
    ], fluid="xl"), className="banda banda-esmalte"),

    html.Section(dbc.Container([
        encabezado("¿Cómo se hizo?", "Tres entregas y nueve pasos. Cada paso usa solo la información que existiría en la práctica en ese "
                                     "momento: nada del conjunto de prueba se filtra hacia el entrenamiento.", etiqueta="Metodología", oscuro=True),
        html.Div([html.Div(className="ruta-linea d-none d-xl-block", **{"aria-hidden": "true"})]
                 + [estacion(i + 1, t, d, e) for i, (t, d, e) in enumerate(RUTA)], className="ruta"),
    ], fluid="xl"), className="banda banda-oscura con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.08"}),

    html.Section(dbc.Container([
        encabezado("¿A qué conclusión se llegó?", "El resumen honesto de los tres entregables. El detalle está en la pestaña del modelo original.",
                   etiqueta="La respuesta corta"),
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
])
