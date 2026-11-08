from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from . import plots
from .data import ADICIONALES, cargar_vinos, json_app, metadata_final, tabla_app
from .tablero import encabezado, fila, lectura, numero, pendiente, tarjeta, termino, unir

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
    ("bi-bar-chart-steps", "Mira toda la nota", "En lugar de la etiqueta bueno / no bueno, cada árbol aprende las probabilidades acumuladas P(nota ≥ 4), …, P(nota ≥ 9)."),
    ("bi-bullseye", "Se enfoca en el corte", "Al partir, pondera esos umbrales con un núcleo centrado en el 7: separar un 6 de un 7 importa más que separar un 3 de un 4."),
    ("bi-diagram-3", "Combina a los vecinos", "Para puntuar un vino suma la probabilidad de «≥ 7» con la de «≥ 6» y «≥ 8», con un segundo núcleo."),
    ("bi-sliders", "Se calibra solo", "Con los vinos que cada árbol no vio (fuera de bolsa) convierte el puntaje en una probabilidad que se puede leer literalmente."),
]


def veredicto(v):
    rechazan = [n for n, ok in zip(("Nemenyi", "Wilcoxon", "t corregida", "DeLong"), v["pruebas_que_rechazan"].values()) if ok]
    condiciones = [
        (v["condicion_1_media_mayor_que_todos"], f"AUC medio mayor que los 9 rivales: {v['auc_bof']:.4f} frente a {v['auc_mejor_benchmark']:.4f} de {v['mejor_benchmark']}."),
        (v["condicion_2_tres_pruebas"], f"Al menos 3 de 4 pruebas a favor frente a {v['mejor_benchmark']}: rechazan {len(rechazan)} "
                                        f"({', '.join(rechazan) if rechazan else 'ninguna'})."),
        (v["condicion_3_efecto_no_despreciable"], f"Efecto no despreciable: δ de Cliff = {v['delta_cliff']:.2f} y d pareada = {v['d_cohen_pareada']:.2f} "
                                                  "(basta uno de los dos)."),
        (v["condicion_4_consistencia"], f"Consistencia en otros datos: gana en {v['conjuntos_donde_bof_gana']} de 4 conjuntos y es significativamente peor en "
                                        f"{v['conjuntos_significativamente_peor']}."),
    ]
    lista = html.Ul([html.Li([html.I(className="bi " + ("bi-check-circle-fill" if ok else "bi-x-circle-fill"), **{"aria-hidden": "true"}),
                              html.Span([html.Strong("Cumple. " if ok else "No cumple. "), texto])], className="cumple" if ok else "falla")
                     for ok, texto in condiciones], className="condiciones")
    sello = html.Div("Superior" if v["superior"] else "No superior", className="veredicto-sello " + ("si" if v["superior"] else "no"))
    if v["superior"]:
        explicacion = "Cumple las cuatro condiciones fijadas antes de los experimentos."
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
        explicacion = (inicio + unir(fallas) + ". Un resultado negativo honesto también es un resultado: el mecanismo compite con los mejores "
                       "métodos actuales, sin superarlos con la certeza que exige el protocolo.")
    return html.Div([
        dbc.Row([
            dbc.Col([html.Div("Veredicto pre-registrado", className="etiqueta-seccion", style={"color": "#a9c2ee"}),
                     html.H2("¿Es superior el modelo original?"), html.P(explicacion, className="mb-0", style={"color": "#c6d4ec"})], lg=8),
            dbc.Col(sello, lg=4, class_name="d-flex align-items-center justify-content-lg-end mt-3 mt-lg-0"),
        ]),
        lista,
    ], className="veredicto con-azulejo mb-4 aparecer", role="region", **{"aria-label": "Veredicto de superioridad"})


def nucleo():
    escala = float(cargar_vinos()["quality"].std())
    columnas = [html.Div([html.Div("", className="nucleo-peso"), html.Div(className="nucleo-barra")], className="nucleo-columna") for _ in range(6)]
    return tarjeta("Juega con el núcleo", [
        dcc.Store(id="nucleo-escala", data=escala),
        html.P(["¿Cuánto escucha el modelo a cada pregunta «¿la nota es ≥ t?» para decidir si un vino es bueno? El ", termino("núcleo"),
                " lo decide. Mueve el ancho y el desplazamiento."], className="small"),
        dbc.Row([
            dbc.Col([html.Label("Ancho del núcleo", htmlFor="nucleo-ancho", className="campo-etiqueta"),
                     dcc.Slider(id="nucleo-ancho", min=0, max=8, step=1, value=4, marks={i: a for i, a in enumerate(ANCHOS)}, updatemode="drag")], md=7),
            dbc.Col([html.Label("Desplazamiento", htmlFor="nucleo-delta", className="campo-etiqueta"),
                     dcc.Slider(id="nucleo-delta", min=-1, max=1, step=0.25, value=0, marks={-1: "−1", 0: "0", 1: "+1"}, updatemode="drag")], md=5),
        ], class_name="mb-3"),
        html.Div(columnas, id="nucleo", className="nucleo-columnas", role="img", **{"aria-label": "Peso de cada umbral según el núcleo"}),
        html.Div([html.Div(f"≥ {t}") for t in range(4, 10)], className="nucleo-rotulos"),
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
            html.P([f"Búsqueda del {r['fecha']} en OpenAlex, Semantic Scholar, arXiv y Crossref, más citas hacia atrás y hacia adelante desde "
                    f"{r['semillas']} trabajos cercanos. ", html.Strong("No se encontraron estudios"), " que apliquen a la calidad del vino un bosque que "
                    "parta con una impureza centrada en el umbral, combine umbrales vecinos y calibre fuera de bolsa. Eso no prueba que nadie lo haya "
                    "hecho: la búsqueda no cubrió bases de pago y se debe repetir antes de la entrega final. Cada pieza por separado sí existe; lo "
                    "nuevo es la combinación para un evento definido por el corte de una nota ordinal."]),
            html.Details([html.Summary("Ver los trabajos más cercanos y en qué se diferencian"), html.Div(tabla, className="mt-2")], className="plegable")]


def adicionales():
    t = tabla_app("adicionales_auc.csv")
    p = tabla_app("adicionales_pruebas.csv")
    if t is None or p is None:
        return pendiente("La comparación en conjuntos adicionales")
    mejores = dict(zip(p["conjunto"], p["mejor_benchmark"]))
    filas = []
    for f in p.itertuples():
        peor = "sí" if f.significativamente_peor else "no"
        filas.append(html.Tr([html.Td(CONJUNTOS.get(f.conjunto, f.conjunto)), html.Td(f"{f.auc_bof:.4f}"), html.Td(f"{f.mejor_benchmark} ({f.auc_mejor:.4f})"),
                              html.Td(f"{int(f.puesto_bof)} de 10"), html.Td("< 0.0001" if f.delong_p_holm < 1e-4 else f"{f.delong_p_holm:.4f}"), html.Td(peor)]))
    tabla = dbc.Table([html.Thead(html.Tr([html.Th(c) for c in ("Conjunto", "AUC del BOF", "Mejor rival", "Puesto", "DeLong (Holm)", "¿Significativamente peor?")])),
                       html.Tbody(filas)], size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")
    puestos = p["puesto_bof"].astype(int).tolist()
    peores = [CONJUNTOS.get(c, c) for c in p.loc[p["significativamente_peor"], "conjunto"]]
    gana = sum(1 for x in puestos if x == 1)
    cierre = (f"Es significativamente peor que el mejor rival en {unir(peores)}." if peores else
              "En ningún conjunto es significativamente peor que el mejor rival.")
    texto = (f"En los cuatro problemas el evento también es el corte de una respuesta ordinal (edad de un abulón, estado fetal, nota de un estudiante, "
             f"gravedad cardiaca). El modelo original queda en los puestos {', '.join(map(str, puestos))} de 10 y es el mejor en {gana} de 4. " + cierre)
    return [plots.figura(plots.adicionales(t, CONJUNTOS, ORIGINAL, mejores)), tabla, lectura(texto)]


def lectura_pruebas():
    pr = tabla_app("pruebas_bof.csv")
    if pr is None:
        return None
    confirman = (pr["wilcoxon_p_holm"] < 0.05) & (pr["t_p_holm"] < 0.05) & (pr["delong_p_holm"] < 0.05) & (pr["diferencia_media"] > 0)
    dudosos = pr.loc[pr["t_p_holm"] >= 0.05, "benchmark"].tolist()
    texto = f"Frente a {int(confirman.sum())} de los 9 rivales, las tres pruebas pareadas confirman la ventaja del modelo original. "
    if dudosos:
        texto += (f"Frente a {unir(dudosos)}, la t corregida, la más prudente porque tiene en cuenta que los bloques comparten vinos, "
                  "no ve una diferencia real. ")
    return lectura([texto + "Los p-valores están ajustados por ", termino("Holm"), "; el tamaño del efecto es el ", termino("delta de Cliff"), "."])


def pruebas_todos():
    pr = tabla_app("pruebas_bof.csv")
    if pr is None:
        return pendiente("Las pruebas de hipótesis")
    p_txt = lambda x: "< 0.0001" if x < 1e-4 else f"{x:.4f}"
    filas = [html.Tr([html.Td(f.benchmark), html.Td(f"{f.diferencia_media:+.4f}"), html.Td(f"{int(f.bloques_a_favor)}/25"), html.Td(p_txt(f.wilcoxon_p_holm)),
                      html.Td(p_txt(f.t_p_holm)), html.Td(p_txt(f.delong_p_holm)), html.Td(f"{f.delta_cliff:+.2f} ({f.magnitud_cliff})")])
             for f in pr.itertuples()]
    return dbc.Table([html.Thead(html.Tr([html.Th(c) for c in ("Rival", "ΔAUC medio", "Bloques a favor", "Wilcoxon (Holm)", "t corregida (Holm)",
                                                                "DeLong (Holm)", "δ de Cliff")])), html.Tbody(filas)],
                     size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")


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


def lectura_ablacion(a):
    """Conclusión de la ablación en el conjunto del vino, escrita a partir de los datos."""
    pruebas = ["wilcoxon_p_holm", "t_p_holm", "delong_p_holm"]
    resto = a[~a["variante"].isin(["completo", "sin_calibracion"])].copy()
    resto["rechazos"] = (resto[pruebas] < 0.05).sum(axis=1)
    frases = ["Cada barra quita una pieza y deja las demás igual, con los mismos hiperparámetros y bloques."]
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
    return " ".join(frases)


def lectura_sensibilidad(sens, bof, rf):
    """Conclusión del análisis de sensibilidad del modelo original frente a Random Forest."""
    frases = ["Cada gráfica mueve un solo hiperparámetro y deja los demás en sus valores finales; la línea ocre es Random Forest en los mismos pliegues."]
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
    return " ".join(frases)


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
    tabla = dbc.Table.from_dataframe(resumen.round(3), size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")
    return [dbc.Row(figs),
            lectura(lectura_sensibilidad(sens, bof, rf)),
            html.H3("Resumen de robustez por modelo", className="tarjeta-titulo mt-4"), tabla,
            lectura("Semillas, ruido, tamaño de muestra, cambio de tintos a blancos y cobertura conforme, medidos con los hiperparámetros finales en "
                    "los 5 pliegues de la repetición 1. " + lectura_resumen(resumen, mejor))]


def convergencia():
    meta = metadata_final("bosque_ordinal_focalizado")
    if not meta or meta.get("metricas_bucle_externo", {}).get("simulado") or "reduccion_arboles" not in meta:
        return pendiente("La curva de convergencia del modelo final")
    curva = pd.DataFrame(meta["reduccion_arboles"]["curva_auc_fuera_de_bolsa"], columns=["arboles", "auc"]).assign(serie="AUC fuera de bolsa")
    fig = plots.lineas(curva, "arboles", "auc", "serie", "¿Cuántos árboles hacen falta?", "Número de árboles", "AUC fuera de bolsa", resaltar="AUC fuera de bolsa")
    return [plots.figura(fig), lectura(f"El AUC fuera de bolsa se estabiliza pronto; el modelo desplegado usa {meta['reduccion_arboles']['arboles_desplegados']} "
                                       "árboles, el menor número con AUC a menos de 0.001 del bosque completo.")]


def contenido():
    v = json_app("veredicto.json")
    if v is None:
        return pendiente("El veredicto del modelo original")
    pasos = [dbc.Col(html.Div([html.I(className=f"bi {icono} fs-3", style={"color": "#1d5fc4"}, **{"aria-hidden": "true"}), html.H3(t, className="tarjeta-titulo mt-2"),
                               html.P(d, className="small mb-0")], className="tarjeta voltea", style={"--i": i}), sm=6, class_name="mb-3")
             for i, (icono, t, d) in enumerate(PASOS)]
    return html.Div([
        veredicto(v),
        encabezado("¿Qué limitación resuelve?", "Todos los demás modelos aprenden con la etiqueta bueno / no bueno: un vino de 6, que estuvo a un punto, "
                                                 "cuenta igual que uno de 3. El Bosque Ordinal Focalizado usa la nota completa, pero sin perder de vista el corte de 7.",
                   etiqueta="La idea"),
        fila(nucleo(), tarjeta("Cómo funciona, en cuatro pasos", [dbc.Row(pasos, class_name="g-2"),
                                                                    html.Details([html.Summary("Ver el pseudocódigo"),
                                                                                  html.Pre(PSEUDOCODIGO, className="pseudocodigo mt-2")], className="plegable")]),
             anchos=[5, 7], clase="g-4 mb-4"),
        encabezado("¿Es realmente nuevo?", "La novedad no se declara: se buscó evidencia de que no existe.", etiqueta="Revisión de literatura"),
        tarjeta(None, novedad(), clase="mb-4"),
        encabezado("¿Funciona también en otros problemas?", "Cuatro conjuntos públicos elegidos antes de evaluar, con el mismo procedimiento.",
                   etiqueta="Conjuntos adicionales"),
        tarjeta(None, adicionales(), clase="mb-4"),
        encabezado("¿Frente a quién gana y frente a quién no?", "Las cuatro pruebas pareadas contra cada rival, con p-valores ajustados por Holm.",
                   etiqueta="Pruebas de hipótesis"),
        tarjeta(None, [pruebas_todos(), lectura_pruebas()], clase="mb-4"),
        encabezado("¿Cada pieza aporta y aguanta cambios?", etiqueta="Ablación y robustez"),
        fila(tarjeta(None, ablacion()), tarjeta(None, convergencia()), clase="g-4 mb-4"),
        tarjeta(None, robustez(), clase="mb-4"),
        html.P([html.I(className="bi bi-file-earmark-text me-2"), "La ficha del modelo (uso previsto, datos, métricas, limitaciones y riesgos) está en ",
                dcc.Link("Ficha del modelo", href="/ficha"), "."]),
    ])
