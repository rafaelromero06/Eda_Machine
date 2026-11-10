import dash
import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Input, Output, clientside_callback, dcc, html

from src.data import NUMERICAS_VINO, cargar_vinos, json_app
from src.tablero import encabezado, idea, lectura, pendiente, siguiente, termino

dash.register_page(__name__, path="/cata", name="Cata a ciegas", title="Cata a ciegas · Cata Ciega", order=3)


def pistas():
    """Sentido en que cada medición difiere en los vinos buenos (+1 más alta, −1 más baja, 0 casi igual) y su fuerza."""
    d = cargar_vinos()
    bueno = d["quality"] >= 7
    dif = (d.loc[bueno, NUMERICAS_VINO].mean() - d.loc[~bueno, NUMERICAS_VINO].mean()) / d[NUMERICAS_VINO].std()
    fuertes = set(dif.abs().sort_values(ascending=False).index[:2])
    return [{"sentido": 0 if abs(dif[c]) < 0.15 else (1 if dif[c] > 0 else -1), "clave": c in fuertes} for c in NUMERICAS_VINO]


def layout():
    datos = json_app("cata.json")
    juego = pendiente("El juego") if datos is None else html.Div(id="juego", className="juego", **{"aria-live": "polite"})
    return html.Div([
        html.Section(dbc.Container([
            html.Div("Juego", className="antetitulo"),
            html.H1("Cata a ciegas: ¿cuál de los dos es el bueno?"),
            html.P("En cada ronda hay dos vinos reales con sus mediciones: uno obtuvo 7 o más en la cata y el otro no. Elige el bueno y mira qué "
                   "eligió el modelo original. Son diez parejas y no hace falta saber de vinos: activa las pistas."),
        ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
        html.Section(dbc.Container([
            dcc.Store(id="juego-datos", data=None if datos is None else {**datos, "pistas": pistas()}),
            html.Div([dbc.Switch(id="juego-pistas", label="Mostrar pistas", value=False, class_name="mb-0"),
                      html.Span("Las pistas marcan qué suele ser más alto (▲) o más bajo (▼) en los vinos buenos; las dos más útiles van resaltadas.",
                                className="small text-secondary")], className="juego-ayuda"),
            juego,
            dbc.Row([
                dbc.Col(html.Div([encabezado("¿Qué tiene que ver esto con el AUC?", etiqueta="Lo que enseña el juego"),
                                  html.P(["Elegir el bueno entre dos vinos es exactamente lo que mide el ", termino("AUC"), ": en cuántas de cada "
                                          "100 parejas el modelo le da más puntaje al vino bueno. 50 de 100 sería lanzar una moneda y 100 de 100, "
                                          "no fallar nunca."]),
                                  lectura(idea("El juego es limpio: el modelo no había visto estos vinos.",
                                               "Sus probabilidades vienen de la validación, donde cada vino lo predijo un modelo que no lo usó "
                                               "para aprender. Al responder, cada copa se llena con la probabilidad que el modelo le dio."))]),
                        lg=7, class_name="mt-5"),
            ]),
        ], fluid="xl"), className="banda banda-compacta"),
        siguiente("/cata", "Para terminar: para qué sirve el modelo, qué no debe hacerse con él y de dónde vienen los datos."),
    ])


clientside_callback(ClientsideFunction("cata", "juego"), Output("juego", "title"), Input("juego-datos", "data"))
