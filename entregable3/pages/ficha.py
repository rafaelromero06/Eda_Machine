import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data import RUTA_BASE
from src.tablero import pendiente

dash.register_page(__name__, path="/ficha", name="Ficha del modelo", title="Ficha del modelo · Cata Ciega", order=4)


def _documento(nombre, que):
    ruta = RUTA_BASE / nombre
    if not ruta.exists():
        return pendiente(que)
    return dcc.Markdown(ruta.read_text(encoding="utf-8"), link_target="_blank", className="documento")


def layout():
    return html.Div([
        html.Section(dbc.Container([
            html.Div("Documentación responsable", className="antetitulo"),
            html.H1("Ficha del modelo y declaración de los datos"),
            html.P("Para qué sirve el Bosque Ordinal Focalizado, cómo se evaluó, qué no debe hacerse con él y de dónde vienen los datos."),
        ], fluid="xl"), className="pagina-titulo con-azulejo", style={"--patron": "#4c86e0", "--opacidad": "0.1"}),
        html.Section(dbc.Container([
            dbc.Tabs([
                dbc.Tab(html.Div(_documento("MODEL_CARD.md", "La ficha del modelo final"), className="tarjeta mt-3"), label="Ficha del modelo (model card)"),
                dbc.Tab(html.Div(_documento("data/DATA_STATEMENT.md", "La declaración de los datos"), className="tarjeta mt-3"),
                        label="Declaración de los datos"),
            ], class_name="documento-tabs"),
        ], fluid="xl"), className="banda banda-compacta"),
    ])
