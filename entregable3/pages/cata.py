import dash
import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Input, Output, clientside_callback, dcc, html

from src.data import json_app
from src.tablero import encabezado, lectura, pendiente, termino

dash.register_page(__name__, path="/cata", name="Cata a ciegas", title="Cata a ciegas · Cata Ciega", order=3)


def layout():
    datos = json_app("cata.json")
    juego = pendiente("El juego") if datos is None else html.Div(id="juego", className="juego", **{"aria-live": "polite"})
    return html.Div([
        html.Section(dbc.Container([
            html.Div("Juego", className="antetitulo"),
            html.H1("Cata a ciegas: ¿cuál de los dos es el bueno?"),
            html.P("En cada ronda aparecen dos vinos reales con su química: uno obtuvo 7 o más en la cata y el otro no. Elige el bueno y luego mira qué "
                   "eligió el modelo original. Son diez parejas."),
        ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
        html.Section(dbc.Container([
            dcc.Store(id="juego-datos", data=datos),
            juego,
            dbc.Row([
                dbc.Col(html.Div([encabezado("¿Qué tiene que ver esto con el AUC?", etiqueta="Lo que enseña el juego"),
                                  html.P(["Elegir el mejor de dos vinos es exactamente lo que mide el ", termino("AUC"), ": la probabilidad de que el "
                                          "modelo le dé más puntaje a un vino bueno que a uno que no lo es. Un AUC de 0.5 es lanzar una moneda; 1 "
                                          "sería no fallar nunca."]),
                                  lectura("Las probabilidades del modelo vienen de la validación: cada vino lo predijo un modelo que no lo había visto, "
                                          "así que el juego es limpio. La copa se llena con la probabilidad que el modelo le dio a cada vino.")]),
                        lg=7, class_name="mt-5"),
            ]),
        ], fluid="xl"), className="banda banda-compacta"),
    ])


clientside_callback(ClientsideFunction("cata", "juego"), Output("juego", "title"), Input("juego-datos", "data"))
