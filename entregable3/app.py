import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from src.tablero import AUTORES, PAGINAS, VERSION, panel_glosario

FUENTES = ("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..600"
           "&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:wght@400..700&display=swap")

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, FUENTES],
    title="Cata Ciega · Calidad del vino con Machine Learning",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": "Dashboard que anticipa la calidad de un vino Vinho Verde a partir de su química"},
               {"name": "theme-color", "content": "#0e2a5c"}],
)
server = app.server

enlaces = [dbc.NavItem(dbc.NavLink([html.I(className=f"bi {icono} me-1", **{"aria-hidden": "true"}), nombre], href=ruta, active="exact"))
           for ruta, nombre, icono, _ in PAGINAS]
boton_glosario = dbc.NavItem(html.Button([html.I(className="bi bi-book me-1", **{"aria-hidden": "true"}), "Glosario"], id="glosario-boton",
                                         type="button", className="nav-link boton-glosario", title="Guía rápida y definiciones en lenguaje sencillo"))

barra = dbc.Navbar(
    dbc.Container([
        dcc.Link([html.Span(className="marca-azulejo", **{"aria-hidden": "true"}), "Cata Ciega"], href="/", className="marca"),
        dbc.NavbarToggler(id="menu-boton"),
        dbc.Collapse(dbc.Nav(enlaces + [boton_glosario], className="ms-auto", navbar=True), id="menu", navbar=True),
    ], fluid="xl"),
    class_name="barra", dark=True, sticky="top",
)

pie = html.Footer([
    html.Div(className="friso", **{"aria-hidden": "true"}),
    html.Div(dbc.Container(dbc.Row([
        dbc.Col([html.Strong("Cata Ciega"), html.Span(f" · {AUTORES} · Machine Learning, Universidad del Norte")], md=8),
        dbc.Col(f"Entregable 3 · versión {VERSION}", md=4, class_name="text-md-end"),
    ]), fluid="xl"), className="pie"),
])

app.layout = html.Div([
    html.A("Saltar al contenido", href="#contenido", className="saltar"),
    barra,
    html.Main(dash.page_container, id="contenido", tabIndex=-1),
    pie,
    dbc.Offcanvas(panel_glosario(), id="glosario", title="Guía rápida y glosario", placement="end", is_open=False, scrollable=True,
                  class_name="panel-glosario"),
    html.Div(id="telon", **{"aria-hidden": "true"}),
])


@app.callback(Output("menu", "is_open"), Input("menu-boton", "n_clicks"), Input("_pages_location", "pathname"),
              State("menu", "is_open"), prevent_initial_call=True)
def alternar_menu(n, ruta, abierto):
    return (not abierto) if dash.ctx.triggered_id == "menu-boton" else False


@app.callback(Output("glosario", "is_open"), Input("glosario-boton", "n_clicks"), State("glosario", "is_open"), prevent_initial_call=True)
def alternar_glosario(n, abierto):
    return not abierto


if __name__ == "__main__":
    app.run(debug=False, port=8050)
