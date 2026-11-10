from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from . import plots
from .data import ADICIONALES, cargar_vinos, json_app, metadata_final, tabla_app
from .tablero import como_leer, encabezado, fila, idea, indice, lectura, numero, p_valor, pendiente, tabla, tarjeta, termino, unir

ORIGINAL = "Bosque Ordinal Focalizado"
CONJUNTOS = {c: d["titulo"].split(" (")[0] for c, d in ADICIONALES.items()}
ANCHOS = ["0", "0.25", "0.5", "0.75", "1", "1.5", "2.5", "5", "∞"]
PARAMETROS = {"ancho_agregacion": "Ancho de agregación", "ancho_particion": "Ancho de partición", "desplazamiento": "Desplazamiento del centro",
              "max_features": "Atributos candidatos por corte", "min_samples_leaf": "Vinos mínimos por hoja", "n_estimators": "Número de árboles"}

PSEUDOCODIGO = """Entrada: vinos (x_i, y_i) con y ordinal; umbral del evento τ = 7; T árboles; m atributos candidatos;
         hoja mínima ℓ; anchos h_p (partición) y h_a (agregación); desplazamiento δ.
Salida:  probabilidad calibrada p(x) = P(y >= τ | x).

1. Umbrales t_1 < ... < t_K (niveles de y salvo el menor). Indicadores Z_ik = 1{y_i >= t_k}.
2. Escala s = desviación estándar de y; centro c = τ + δ·s.
   Pesos de partición  w_k ∝ exp(-(t_k - c)² / (2 h_p² s²));  pesos de agregación ω_k ∝ exp(-(t_k - c)² / (2 h_a² s²)).
3. Para cada árbol b = 1..T:
     a. Muestra bootstrap de tamaño n.
     b. Crecer por niveles. En cada nodo S con al menos 2ℓ casos y G(S) > 0:
          G(S) = Σ_k w_k F_k(S) (1 - F_k(S)),   F_k(S) = proporción ponderada de casos con y >= t_k.
          Elegir m atributos al azar y un corte uniforme en [mín, máx] de cada uno.
          Ganancia Δ = n_S G(S) - n_L G(L) - n_R G(R); dividir por el candidato de mayor Δ.
        Cada hoja guarda la distribución acumulada F(hoja).
4. Puntaje: s(x) = (1/T) Σ_b Σ_k ω_k F_k(hoja_b(x)).
5. Calibración: con los puntajes fuera de bolsa, ajustar (a, b) por Newton (Platt).
6. Devolver p(x) = σ(a + b·logit s(x)) y la distribución acumulada F(x)."""

PASOS = [
    ("bi-bar-chart-steps", "Usa toda la nota", "En vez de aprender solo «bueno / no bueno», cada árbol aprende seis respuestas a la vez: ¿el vino llega a 4?, ¿a 5?, …, ¿a 9?"),
    ("bi-bullseye", "Se concentra en el 7", "Al dividir los vinos da más peso a las preguntas cercanas al corte: separar un 6 de un 7 importa más que separar un 3 de un 4."),
    ("bi-diagram-3", "Escucha a los vecinos", "Para puntuar un vino combina «¿llega a 7?» con «¿llega a 6?» y «¿llega a 8?», que traen información parecida."),
    ("bi-sliders", "Se corrige solo", "Con los vinos que cada árbol no usó para aprender ajusta su puntaje, para que se pueda leer como una probabilidad."),
]
SECCIONES = [("sec-veredicto", "Veredicto"), ("sec-idea", "La idea"), ("sec-novedad", "¿Es nuevo?"), ("sec-otros", "Otros problemas"),
             ("sec-pruebas", "Pruebas"), ("sec-piezas", "Piezas y robustez")]
COLUMNAS_ROBUSTEZ = {
    "Modelo": ("Modelo", None),
    "AUC de referencia": ("AUC de referencia", "AUC en los 5 pliegues de la repetición 1, sin alterar nada"),
    "DE entre 10 semillas": ("Cambio entre 10 semillas", "Cuánto varía el AUC al cambiar el azar del entrenamiento. Vacío: el modelo no usa azar"),
    "Caída con ruido σ = 0.25": ("Caída con ruido", "AUC que pierde al añadir ruido a las mediciones (0.25 desviaciones estándar)"),
    "Caída con 10 % de etiquetas movidas": ("Caída con notas alteradas", "AUC que pierde al mover un punto la nota del 10 % de los vinos"),
    "Caída con 5 % de atípicos": ("Caída con atípicos", "AUC que pierde al contaminar el 5 % de los vinos con valores extremos"),
    "AUC con 25 % de los datos": ("AUC con 1 de cada 4 vinos", "AUC al entrenar solo con el 25 % de los vinos"),
    "AUC tinto → blanco": ("Tinto → blanco", "AUC al entrenar solo con tintos y evaluar en blancos"),
    "AUC blanco → tinto": ("Blanco → tinto", "AUC al entrenar solo con blancos y evaluar en tintos"),
    "Cobertura conforme 90 %": ("Garantía del 90 %", "Porcentaje de vinos en que el conjunto conforme al 90 % acertó"),
    "Tamaño medio del conjunto": ("Respuestas por vino", "Tamaño medio del conjunto conforme: 1 es seguro y 2 es que no decide"),
    "AUC con bloques del archivo": ("AUC por lotes", "AUC con pliegues de vinos consecutivos del archivo, que podrían ser del mismo lote"),
}


def veredicto(v):
    rechazan = [n for n, ok in zip(("Nemenyi", "Wilcoxon", "t corregida", "DeLong"), v["pruebas_que_rechazan"].values()) if ok]
    condiciones = [
        (v["condicion_1_media_mayor_que_todos"], "Tener el mejor promedio",
         f"AUC medio de {v['auc_bof']:.4f} frente a {v['auc_mejor_benchmark']:.4f} de {v['mejor_benchmark']}, el mejor de los 9 rivales."),
        (v["condicion_2_tres_pruebas"], "Que 3 de 4 pruebas confirmen la ventaja",
         f"Frente a {v['mejor_benchmark']} la confirman {len(rechazan)}" + (f" ({unir(rechazan)})." if rechazan else ".")),
        (v["condicion_3_efecto_no_despreciable"], "Que la ventaja no sea diminuta",
         f"δ de Cliff = {v['delta_cliff']:.2f} y d pareada = {v['d_cohen_pareada']:.2f}; bastaba con que uno de los dos no fuera despreciable."),
        (v["condicion_4_consistencia"], "Repetirlo en otros datos",
         f"Gana en {v['conjuntos_donde_bof_gana']} de 4 conjuntos y es claramente peor en {v['conjuntos_significativamente_peor']}."),
    ]
    lista = html.Ul([html.Li([html.I(className="bi " + ("bi-check-circle-fill" if ok else "bi-x-circle-fill"), **{"aria-hidden": "true"}),
                              html.Span([html.Strong(f"{titulo}: {'cumple' if ok else 'no cumple'}. "), texto])], className="cumple" if ok else "falla")
                     for ok, titulo, texto in condiciones], className="condiciones")
    sello = html.Div("Superior" if v["superior"] else "No superior", className="veredicto-sello " + ("si" if v["superior"] else "no"))
    if v["superior"]:
        explicacion = "Cumple las cuatro condiciones que se fijaron antes de los experimentos."
    else:
        fallas = []
        if not v["condicion_1_media_mayor_que_todos"]:
            fallas.append("no tiene el AUC medio más alto")
        if not v["condicion_2_tres_pruebas"]:
            fallas.append(f"solo {len(rechazan)} de 4 pruebas confirman la ventaja")
        if not v["condicion_3_efecto_no_despreciable"]:
            fallas.append("el tamaño del efecto es despreciable")
        if not v["condicion_4_consistencia"]:
            gana = v["conjuntos_donde_bof_gana"]
            fallas.append("no gana en ninguno de los 4 conjuntos adicionales" if gana == 0 else f"solo gana en {gana} de los 4 conjuntos adicionales")
        inicio = "Tiene el AUC medio más alto de los 10 modelos, pero " if v["condicion_1_media_mayor_que_todos"] else "No es superior: "
        explicacion = (inicio + unir(fallas) + ". Para llamarlo superior debía cumplir cuatro condiciones escritas antes de los experimentos, "
                       "y cumple " + str(sum(bool(v[c]) for c in ("condicion_1_media_mayor_que_todos", "condicion_2_tres_pruebas",
                                                                     "condicion_3_efecto_no_despreciable", "condicion_4_consistencia")))
                       + ". Compite con los mejores métodos actuales, sin superarlos con esa certeza.")
    return html.Div([
        dbc.Row([
            dbc.Col([html.Div("Veredicto pre-registrado", className="etiqueta-seccion", style={"color": "#a9c2ee"}),
                     html.H2("¿Es superior el modelo original?"), html.P(explicacion, className="mb-0", style={"color": "#c6d4ec"})], lg=8),
            dbc.Col(sello, lg=4, class_name="d-flex align-items-center justify-content-lg-end mt-3 mt-lg-0"),
        ]),
        lista,
    ], className="veredicto con-azulejo mb-4 aparecer", role="region", id="sec-veredicto", **{"aria-label": "Veredicto de superioridad"})


def nucleo():
    escala = float(cargar_vinos()["quality"].std())
    columnas = [html.Div([html.Div("", className="nucleo-peso"), html.Div(className="nucleo-barra")], className="nucleo-columna") for _ in range(6)]
    return tarjeta("¿A qué preguntas presta atención?", [
        dcc.Store(id="nucleo-escala", data=escala),
        html.P(["Para decidir si un vino llega a 7, el modelo combina seis preguntas: ¿llega a 4?, ¿a 5?, …, ¿a 9? El ", termino("núcleo"),
                " reparte la atención entre ellas. Mueve el ancho: con 0 solo escucha «¿llega a 7?» y con ∞ las escucha a todas por igual."],
               className="small"),
        dbc.Row([
            dbc.Col([html.Label("Ancho del núcleo", htmlFor="nucleo-ancho", className="campo-etiqueta"),
                     dcc.Slider(id="nucleo-ancho", min=0, max=8, step=1, value=4, marks={i: a for i, a in enumerate(ANCHOS)}, updatemode="drag")], md=7),
            dbc.Col([html.Label("Desplazamiento", htmlFor="nucleo-delta", className="campo-etiqueta"),
                     dcc.Slider(id="nucleo-delta", min=-1, max=1, step=0.25, value=0, marks={-1: "−1", 0: "0", 1: "+1"}, updatemode="drag")], md=5),
        ], class_name="mb-3"),
        html.Div(columnas, id="nucleo", className="nucleo-columnas", role="img", **{"aria-label": "Peso de cada umbral según el núcleo"}),
        html.Div([html.Div(f"¿llega a {t}?") for t in range(4, 10)], className="nucleo-rotulos"),
        lectura("", id="nucleo-texto"),
    ])


def novedad():
    r = json_app("revision.json")
    rel = tabla_app("trabajos_relacionados.csv")
    if r is None:
        return pendiente("La revisión de literatura")
    cifras = html.Div([html.Div([html.Div(numero(valor), className="cifra-valor"), html.Div(texto, className="cifra-etiqueta")], className="cifra voltea",
                                style={"--i": i})
                       for i, (valor, texto) in enumerate([(r["bases"], "bases bibliográficas"), (r["cadenas"], "cadenas de búsqueda"),
                                                           (r["revisados"], "títulos revisados"), (r["citacion"], "títulos por citación"),
                                                           (r["consultas_github"], "búsquedas en GitHub")])], className="embudo mb-3")
    tabla = None
    if rel is not None:
        tabla = dbc.Table([html.Thead(html.Tr([html.Th("Método"), html.Th("Qué predice"), html.Th("Usa la nota ordinal"), html.Th("Combina umbrales vecinos"),
                                               html.Th("Qué hace distinto el BOF")])),
                           html.Tbody([html.Tr([html.Td(f.metodo), html.Td(f.objetivo), html.Td(f.informacion_ordinal), html.Td(f.agrega_umbrales),
                                                html.Td(f.diferencia)]) for f in rel.itertuples()])],
                          size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")
    return [cifras,
            html.P([html.Strong("No se encontró ningún estudio que use este mecanismo para la calidad del vino. "),
                    f"La búsqueda se hizo el {r['fecha']} en OpenAlex, Semantic Scholar, arXiv y Crossref, y siguió las citas de {r['semillas']} "
                    "trabajos cercanos. Eso no prueba que nadie lo haya hecho: no cubrió bases de pago y debe repetirse antes de la entrega "
                    "final. Cada pieza por separado ya existía; lo nuevo es combinarlas para un evento definido por el corte de una nota "
                    "ordinal: partir los árboles mirando sobre todo el corte, combinar los umbrales vecinos y calibrar con los vinos fuera de bolsa."]),
            html.Details([html.Summary("Ver los trabajos más cercanos y en qué se diferencian"), html.Div(tabla, className="mt-2")], className="plegable")]


def adicionales():
    t = tabla_app("adicionales_auc.csv")
    p = tabla_app("adicionales_pruebas.csv")
    if t is None or p is None:
        return pendiente("La comparación en conjuntos adicionales")
    mejores = dict(zip(p["conjunto"], p["mejor_benchmark"]))
    filas = [(CONJUNTOS.get(f.conjunto, f.conjunto), f"{f.auc_bof:.4f}", f"{f.mejor_benchmark} ({f.auc_mejor:.4f})", f"{int(f.puesto_bof)} de 10",
              p_valor(f.delong_p_holm), "sí" if f.significativamente_peor else "no") for f in p.itertuples()]
    tabla_otros = tabla([("Conjunto de datos", None), ("AUC del modelo original", None), ("Mejor rival (su AUC)", None), ("Puesto", None),
                         ("p ajustado", "Prueba de DeLong frente al mejor rival, con ajuste de Holm. Menor que 0.05: la diferencia es real"),
                         ("¿Claramente peor?", "Sí cuando el mejor rival le gana con p ajustado menor que 0.05")], filas)
    puestos = p["puesto_bof"].astype(int).tolist()
    peores = [CONJUNTOS.get(c, c) for c in p.loc[p["significativamente_peor"], "conjunto"]]
    gana = sum(1 for x in puestos if x == 1)
    cierre = (f"Solo en {unir(peores)} el mejor rival le gana con claridad estadística." if peores else
              "En ninguno el mejor rival le gana con claridad estadística.")
    titular = ("Compite en todos, pero no gana en ninguno." if gana == 0 else f"Gana en {gana} de los 4 problemas.")
    texto = idea(titular, f"Queda en los puestos {unir(map(str, puestos))} de 10. Son problemas del mismo tipo que el del vino: la respuesta tiene "
                          "orden (edad de un abulón, estado fetal, nota de un estudiante, gravedad cardiaca) y se pregunta si pasa de un corte. "
                          + cierre)
    return [plots.figura(plots.adicionales(t, CONJUNTOS, ORIGINAL, mejores)),
            como_leer("cada fila es un problema distinto. El punto azul es el modelo original; el rombo ocre, el mejor rival; los grises, los demás. "
                      "Más a la derecha, mejor."),
            tabla_otros, lectura(texto)]


def lectura_pruebas():
    pr = tabla_app("pruebas_bof.csv")
    if pr is None:
        return None
    confirman = (pr["wilcoxon_p_holm"] < 0.05) & (pr["t_p_holm"] < 0.05) & (pr["delong_p_holm"] < 0.05) & (pr["diferencia_media"] > 0)
    dudosos = pr.loc[pr["t_p_holm"] >= 0.05, "benchmark"].tolist()
    n = int(confirman.sum())
    titular = (f"Les gana con claridad a {n} de los 9 rivales; con {unir(dudosos)} hay empate técnico." if dudosos
               else f"Les gana con claridad a {n} de los 9 rivales.")
    detalle = ("Hay empate cuando la t corregida, la prueba más prudente porque tiene en cuenta que los exámenes comparten vinos, no ve una "
               "diferencia real. " if dudosos else "")
    return lectura(idea(titular, detalle + "Un ", termino("p-valor"), " menor que 0.05 indica una diferencia real. Están ajustados por ",
                        termino("Holm"), " y el tamaño del efecto es el ", termino("delta de Cliff"), "."))


def pruebas_todos():
    pr = tabla_app("pruebas_bof.csv")
    if pr is None:
        return pendiente("Las pruebas de hipótesis")
    filas = [(f.benchmark, f"{f.diferencia_media:+.4f}", f"{int(f.bloques_a_favor)} de 25", p_valor(f.wilcoxon_p_holm), p_valor(f.t_p_holm),
              p_valor(f.delong_p_holm), f"{f.delta_cliff:+.2f} ({f.magnitud_cliff})") for f in pr.itertuples()]
    ayuda = "p-valor ajustado por Holm. Menor que 0.05: la diferencia es real"
    return tabla([("Rival", None), ("Ventaja media de AUC", "AUC del modelo original menos AUC del rival, promediado en los 25 exámenes"),
                  ("Exámenes ganados", "En cuántos de los 25 exámenes el modelo original tuvo más AUC"), ("Wilcoxon", ayuda),
                  ("t corregida", ayuda), ("DeLong", ayuda), ("Tamaño del efecto (δ de Cliff)", "Qué tan seguido supera al rival, de −1 a 1")], filas)


def ablacion():
    a = tabla_app("ablacion.csv")
    if a is None:
        return pendiente("La ablación (quitar cada pieza del modelo para medir cuánto aporta)")
    a = a[a["conjunto"] == "vino"].copy()
    nombres = {"completo": "Modelo completo", "sin_agregacion": "Sin combinar umbrales", "agregacion_uniforme": "Todos los umbrales iguales",
               "particion_binaria": "Partición binaria", "sin_ordinal": "Sin la nota ordinal", "sin_calibracion": "Sin calibración"}
    a["modelo"] = a["variante"].map(nombres)
    fig = plots.barras_modelos(a, "auc_media", "¿Qué pasa si se quita cada pieza?", "AUC medio en 25 bloques", "Modelo completo", error="auc_sd", formato=".4f")
    fig.update_xaxes(range=[float((a["auc_media"] - a["auc_sd"]).min()) - 0.01, float((a["auc_media"] + a["auc_sd"]).max()) + 0.012])
    return [plots.figura(fig), lectura(lectura_ablacion(a))]


def tabla_robustez(resumen):
    """Resumen de robustez con encabezados cortos que se explican al pasar el cursor y la fila del modelo original resaltada."""
    columnas = [COLUMNAS_ROBUSTEZ.get(c, (c, None)) for c in resumen.columns]
    filas = [tuple("" if pd.isna(v) else (v if isinstance(v, str) else f"{v:.3f}") for v in f) for f in resumen.itertuples(index=False)]
    modelos = resumen[resumen.columns[0]].tolist()
    return tabla(columnas, filas, destacar=modelos.index(ORIGINAL) if ORIGINAL in modelos else None)


def lectura_ablacion(a):
    """Conclusión de la ablación en el conjunto del vino, escrita a partir de los datos."""
    pruebas = ["wilcoxon_p_holm", "t_p_holm", "delong_p_holm"]
    resto = a[~a["variante"].isin(["completo", "sin_calibracion"])].copy()
    resto["rechazos"] = (resto[pruebas] < 0.05).sum(axis=1)
    frases = []
    mayor = resto.loc[resto["diferencia_vs_completo"].idxmax()]
    confirmada = mayor["rechazos"] >= 2
    frases.append(f"La caída más grande es la de «{mayor['modelo']}»: el AUC baja {numero(mayor['diferencia_vs_completo'], 4)}"
                  + ("." if confirmada else ", aunque las pruebas pareadas no la confirman (se exigen 2 de 3, con Holm)."))
    otras = resto[(resto["rechazos"] >= 2) & (resto["variante"] != mayor["variante"])]
    if len(otras):
        frases.append("Las pruebas sí confirman la caída de " + unir([f"«{r.modelo}» ({numero(r.diferencia_vs_completo, 4)})" for r in otras.itertuples()]) + ".")
    elif not confirmada:
        frases.append("Ninguna otra caída la confirman las pruebas: cada pieza aporta poco por separado.")
    completo = a[a["variante"] == "completo"]
    sin_cal = a[a["variante"] == "sin_calibracion"]
    if len(completo) and len(sin_cal):
        b0, b1 = float(completo["brier_media"].iloc[0]), float(sin_cal["brier_media"].iloc[0])
        if abs(float(sin_cal["auc_media"].iloc[0]) - float(completo["auc_media"].iloc[0])) < 1e-6 and b1 > b0:
            frases.append(f"Quitar la calibración no cambia el AUC, porque conserva el orden de los vinos, pero el puntaje de Brier empeora de "
                          f"{numero(b0, 4)} a {numero(b1, 4)}: la calibración es lo que permite leer la probabilidad tal cual.")
    confirmadas = int((resto["rechazos"] >= 2).sum())
    titular = ("Ninguna pieza cambia mucho el resultado por sí sola." if confirmadas == 0 else
               f"Quitar piezas baja poco el resultado; solo {'una caída es' if confirmadas == 1 else f'{confirmadas} caídas son'} estadísticamente clara"
               f"{'' if confirmadas == 1 else 's'}.")
    return idea(titular, "Cada barra es el modelo sin una de sus piezas, con todo lo demás igual. " + " ".join(frases))


def lectura_sensibilidad(sens, bof, rf):
    """Conclusión del análisis de sensibilidad del modelo original frente a Random Forest."""
    amplitud = sens.groupby("parametro")["auc"].agg(lambda s: s.max() - s.min())
    sensible = amplitud.idxmax()
    titular = (f"El modelo es estable: lo que más lo mueve es {PARAMETROS.get(sensible, sensible).lower()}, y cambia el AUC en {numero(amplitud.max(), 3)}.")
    frases = ["Cada gráfica mueve un solo ajuste y deja los demás en sus valores finales; la línea ocre es Random Forest en los mismos pliegues."]
    if bof is not None and rf is not None:
        relacion = "por encima de" if bof > rf else "por debajo de"
        frases.append(f"Con los valores finales, el modelo original logra {numero(bof, 4)}, {relacion} Random Forest ({numero(rf, 4)}); "
                      f"lo supera en {int((sens['auc'] > rf).sum())} de {len(sens)} variaciones.")
    mejor = sens.loc[sens["auc"].idxmax()]
    if bof is not None and mejor["auc"] - bof > 0.001:
        valor = {"inf": "∞", "None": "todas"}.get(str(mejor["valor"]), str(mejor["valor"]))
        dentro = mejor["auc"] - bof < mejor["auc_sd"]
        frases.append(f"El mejor resultado ({numero(mejor['auc'], 4)}) aparece con {PARAMETROS.get(mejor['parametro'], mejor['parametro']).lower()} = {valor}; "
                      + ("la diferencia con el valor final es menor que la variación entre pliegues, así que no contradice a la búsqueda, que eligió con otros pliegues."
                         if dentro else "la diferencia con el valor final supera la variación entre pliegues."))
    return idea(titular, " ".join(frases))


def lectura_resumen(resumen, mejor):
    """Conclusiones de la tabla de robustez para el modelo original y el mejor rival."""
    r = resumen.set_index("Modelo")
    if ORIGINAL not in r.index:
        return ""
    o = r.loc[ORIGINAL]
    frases = []
    if "DE entre 10 semillas" in r:
        frases.append(f"Con 10 semillas distintas, el AUC del modelo original varía apenas {numero(o['DE entre 10 semillas'], 4)}.")
    if {"AUC tinto → blanco", "AUC blanco → tinto"} <= set(r.columns):
        rival = f", frente a {numero(r.loc[mejor, 'AUC tinto → blanco'], 3)} y {numero(r.loc[mejor, 'AUC blanco → tinto'], 3)} de {mejor}" if mejor in r.index else ""
        mejores = [m for m in r.index if m != ORIGINAL and (r.loc[m, "AUC tinto → blanco"] > o["AUC tinto → blanco"]
                                                            or r.loc[m, "AUC blanco → tinto"] > o["AUC blanco → tinto"])]
        extra = (f"; solo {unir(mejores)} lo {'supera' if len(mejores) == 1 else 'superan'} en alguna de las dos direcciones" if mejores
                 else "; ningún otro modelo transfiere mejor")
        frases.append(f"Entrenado solo con tintos y evaluado en blancos logra {numero(o['AUC tinto → blanco'], 3)}, y al revés "
                      f"{numero(o['AUC blanco → tinto'], 3)}{rival}{extra}.")
    if {"AUC de referencia", "AUC con bloques del archivo"} <= set(r.columns):
        frases.append(f"Con pliegues por bloques de vinos consecutivos (posibles lotes) baja de {numero(o['AUC de referencia'], 3)} a "
                      f"{numero(o['AUC con bloques del archivo'], 3)}, como el resto de modelos.")
    if "Cobertura conforme 90 %" in r:
        c = r["Cobertura conforme 90 %"]
        frases.append(f"Los conjuntos conformes cubren entre {numero(100 * c.min(), 1)} % y {numero(100 * c.max(), 1)} % de los vinos, "
                      f"cerca del 90 % prometido.")
    return " ".join(frases)


def robustez():
    resumen = tabla_app("robustez_resumen.csv")
    sens = tabla_app("robustez_sensibilidad.csv")
    if resumen is None or sens is None:
        return [pendiente("Los estudios de robustez (sensibilidad, semillas, ruido, tamaño de muestra, cambio de distribución y cobertura conforme)")]
    referencia = tabla_app("robustez_referencia.csv")
    rf = float(referencia.loc[referencia["modelo"] == "Random Forest", "auc"].mean()) if referencia is not None else None
    bof = float(referencia.loc[referencia["modelo"] == ORIGINAL, "auc"].mean()) if referencia is not None else None
    mejor = (json_app("veredicto.json") or {}).get("mejor_benchmark", "CatBoost")
    figs = []
    for p in list(dict.fromkeys(sens["parametro"])):
        d = sens[sens["parametro"] == p].copy()
        d["serie"] = ORIGINAL
        d["x"] = d["valor"].map(lambda s: {"inf": "∞", "None": "todas"}.get(str(s), str(s)))
        nombre = PARAMETROS.get(p, p)
        fig = plots.lineas(d, "x", "auc", "serie", nombre, f"{nombre} ({p})", "AUC medio (5 pliegues)", resaltar=ORIGINAL, ordenar=False,
                           referencia=("Random Forest", rf) if rf else None)
        fig.update_xaxes(type="category")
        fig.update_layout(height=300, showlegend=False)
        figs.append(dbc.Col(plots.figura(fig), lg=4, md=6))
    return [html.H3("¿Qué pasa si se cambia uno de sus ajustes?", className="tarjeta-titulo"), dbc.Row(figs),
            lectura(lectura_sensibilidad(sens, bof, rf)),
            html.H3("¿Aguanta condiciones difíciles?", className="tarjeta-titulo mt-4"),
            html.P("Cada columna es una prueba de esfuerzo. Pasa el cursor por un encabezado para ver en qué consiste; la fila azul es el modelo original.",
                   className="small text-secondary"),
            tabla_robustez(resumen),
            lectura(idea("Así responde el modelo original cuando las condiciones se ponen difíciles.",
                         lectura_resumen(resumen, mejor) + " Todo se midió con los ajustes finales en los 5 pliegues de la repetición 1."))]


def convergencia():
    meta = metadata_final("bosque_ordinal_focalizado")
    if not meta or meta.get("metricas_bucle_externo", {}).get("simulado") or "reduccion_arboles" not in meta:
        return pendiente("La curva de convergencia del modelo final")
    curva = pd.DataFrame(meta["reduccion_arboles"]["curva_auc_fuera_de_bolsa"], columns=["arboles", "auc"]).assign(serie="AUC fuera de bolsa")
    fig = plots.lineas(curva, "arboles", "auc", "serie", "¿Cuántos árboles hacen falta?", "Número de árboles", "AUC fuera de bolsa", resaltar="AUC fuera de bolsa")
    return [plots.figura(fig), lectura(idea(f"Con {meta['reduccion_arboles']['arboles_desplegados']} árboles ya rinde como el bosque completo.",
                                            "Añadir más árboles casi no mejora: se despliega el menor número cuyo AUC queda a menos de 0.001 del "
                                            "bosque entero, medido con los vinos ", termino("fuera de bolsa"), "."))]


def vista_comparada():
    """Lo que ven los modelos habituales (dos cajas) frente a lo que ve el modelo original (la nota completa)."""
    notas = [html.Div(str(n), className="vista-nota " + ("bueno" if n >= 7 else "malo")) for n in range(3, 10)]
    notas.insert(4, html.Div(className="vista-corte", **{"aria-hidden": "true"}))
    return html.Div([
        html.Div([html.Div("Lo que ven los demás modelos", className="vista-titulo"),
                  html.Div([html.Div("No bueno", className="vista-caja malo"), html.Div("Bueno", className="vista-caja bueno")], className="vista-fila dos"),
                  html.Div("Un 6 y un 3 caen en la misma caja.", className="vista-pie")], className="vista"),
        html.Div([html.Div("Lo que ve el modelo original", className="vista-titulo"),
                  html.Div(notas, className="vista-fila siete"),
                  html.Div("Sabe que un 6 está a un paso del 7.", className="vista-pie")], className="vista"),
    ], className="vistas tarjeta aparecer mb-4", role="img",
        **{"aria-label": "Los demás modelos ven dos cajas, bueno y no bueno; el modelo original ve las siete notas, de 3 a 9, con el corte en 7."})


def contenido():
    v = json_app("veredicto.json")
    if v is None:
        return pendiente("El veredicto del modelo original")
    pasos = [dbc.Col(html.Div([html.I(className=f"bi {icono} fs-3", style={"color": "#1d5fc4"}, **{"aria-hidden": "true"}), html.H3(t, className="tarjeta-titulo mt-2"),
                               html.P(d, className="small mb-0")], className="tarjeta voltea", style={"--i": i}), sm=6, class_name="mb-3")
             for i, (icono, t, d) in enumerate(PASOS)]
    return html.Div([
        indice(SECCIONES),
        veredicto(v),
        encabezado("¿Qué limitación resuelve?", "Los demás modelos aprenden solo con «bueno / no bueno»: para ellos un vino de 6, que se quedó a un "
                                                 "punto, vale lo mismo que uno de 3. El Bosque Ordinal Focalizado usa la nota completa sin perder de "
                                                 "vista el corte de 7.", etiqueta="5 · Modelo original · La idea", ancla="sec-idea"),
        vista_comparada(),
        fila(nucleo(), tarjeta("Cómo funciona, en cuatro pasos", [dbc.Row(pasos, class_name="g-2"),
                                                                    html.Details([html.Summary("Ver el pseudocódigo"),
                                                                                  html.Pre(PSEUDOCODIGO, className="pseudocodigo mt-2")], className="plegable")]),
             anchos=[5, 7], clase="g-4 mb-4"),
        encabezado("¿Es realmente nuevo?", "La novedad no se declara: se buscó si alguien ya lo había hecho.", etiqueta="Revisión de literatura",
                   ancla="sec-novedad"),
        tarjeta(None, novedad(), clase="mb-4"),
        encabezado("¿Funciona también en otros problemas?", "Cuatro conjuntos de datos públicos, elegidos antes de evaluar y medidos con el mismo "
                                                             "procedimiento.", etiqueta="Conjuntos adicionales", ancla="sec-otros"),
        tarjeta(None, adicionales(), clase="mb-4"),
        encabezado("¿Frente a quién gana y frente a quién no?", "Tres pruebas pareadas contra cada rival. Los números de las pruebas son p-valores "
                                                                 "ajustados: menos de 0.05 significa que la diferencia es real.",
                   etiqueta="Pruebas de hipótesis", ancla="sec-pruebas"),
        tarjeta(None, [pruebas_todos(), lectura_pruebas()], clase="mb-4"),
        encabezado("¿Cada pieza aporta y aguanta cambios?", ["Dos preguntas: qué pasa si se le quita una pieza (", termino("ablación"),
                                                             ") y qué pasa si cambian las condiciones (", termino("robustez"), ")."],
                   etiqueta="Ablación y robustez", ancla="sec-piezas"),
        fila(tarjeta(None, ablacion()), tarjeta(None, convergencia()), clase="g-4 mb-4"),
        tarjeta(None, robustez(), clase="mb-4"),
        html.P([html.I(className="bi bi-file-earmark-text me-2"), "La ficha del modelo (uso previsto, datos, métricas, limitaciones y riesgos) está en ",
                dcc.Link("Ficha del modelo", href="/ficha"), "."]),
    ])
