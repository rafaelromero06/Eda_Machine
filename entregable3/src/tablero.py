from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

VERSION = "1.1 · noviembre de 2026"
AUTORES = "Roger Daza, Matteo Del Sole y Rafael Romero"
ORIGINAL = "Bosque Ordinal Focalizado"
CORTOS = {"Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM", "KNN": "KNN", "Regresión Logística (L1/L2)": "RL",
          "Decision Tree": "Árbol", "Naive Bayes": "NB", "LightGBM": "LGBM", "CatBoost": "CB", "Bosque Ordinal Focalizado": "BOF"}
NOMBRES = {"Bosque Ordinal Focalizado": "Bosque Ordinal (propio)", "Random Forest": "Random Forest", "XGBoost": "XGBoost", "SVM": "SVM",
           "KNN": "KNN", "Regresión Logística (L1/L2)": "Regresión logística", "Decision Tree": "Árbol de decisión",
           "Naive Bayes": "Naive Bayes", "LightGBM": "LightGBM", "CatBoost": "CatBoost"}
PAGINAS = [
    ("/", "Contexto", "bi-compass", "El problema, los datos y la conclusión"),
    ("/eda", "Datos", "bi-bar-chart-line", "Qué distingue a un vino bueno"),
    ("/modelos", "Modelos", "bi-cpu", "Predice un vino y compara los modelos"),
    ("/cata", "Cata a ciegas", "bi-controller", "Juega contra el modelo"),
    ("/ficha", "Ficha del modelo", "bi-file-earmark-text", "Usos, límites y origen de los datos"),
]
GLOSARIO = {
    "bueno": "En este proyecto, un vino que obtuvo una nota de 7 o más en la cata. Son 19 de cada 100.",
    "probabilidad": "Qué tan seguro está el modelo de que el vino es bueno, de 0 % a 100 %. Si dice 70 %, de cada 100 vinos parecidos unos 70 "
                    "deberían ser buenos.",
    "ordinal": "Con orden: una nota de 6 está más cerca de 7 que una de 3. Reducir la nota a «bueno / no bueno» borra esa información.",
    "mediana": "El valor del medio: la mitad de los vinos está por debajo y la otra mitad por encima.",
    "desviación estándar": "Cuánto se alejan los valores de su promedio. Sirve para comparar mediciones que usan unidades distintas.",
    "correlación": "Número entre −1 y 1. Positivo: las dos mediciones suben juntas. Negativo: una baja cuando la otra sube. Cerca de 0: no se "
                   "relacionan.",
    "atípico": "Valor muy alejado de lo habitual. Aquí se usa la regla de las cajas: más de 1.5 veces el rango intercuartílico (RIC) por fuera "
               "de la caja.",
    "prevalencia": "Proporción de vinos buenos en los datos: 19 %. Es lo que acertaría quien marcara vinos al azar.",
    "benchmark": "Modelo conocido que sirve de referencia para comparar. Aquí hay 9.",
    "balanceo": "Técnicas para que el modelo no ignore a los vinos buenos, que son minoría: darles más peso o crear ejemplos sintéticos "
                "(SMOTE, ADASYN).",
    "optimizador": "Método que busca la mejor configuración de un modelo probando varias: en rejilla (Grid), al azar (Random), bayesiana o "
                   "genética.",
    "hiperparámetro": "Ajuste de un modelo que se fija antes de entrenarlo, como el número de árboles.",
    "validación anidada": "Dos bucles: el interno elige los hiperparámetros y el externo mide el desempeño con vinos que el modelo nunca vio.",
    "bloque": "Un examen: el modelo se entrena con una parte de los vinos y se evalúa con otra que no vio. Hay 25: 5 repeticiones de 5 pliegues.",
    "pliegue": "Una de las 5 partes en que se dividen los vinos. El modelo se entrena con 4 y se examina con la que queda.",
    "fuera de muestra": "Predicción de un vino hecha por un modelo que no lo usó para aprender. Es la única forma honesta de medir.",
    "AUC": "Si tomas al azar un vino bueno y uno que no lo es, es la probabilidad de que el modelo le dé más puntaje al bueno. 0.5 es azar y 1 es "
           "perfecto.",
    "PR-AUC": "Mide qué tan limpia es la lista de vinos que el modelo marca como buenos. Es útil porque los vinos buenos son minoría.",
    "Brier": "Error cuadrático de las probabilidades: compara lo que dijo el modelo con lo que pasó. Mientras más bajo, mejor.",
    "F1": "Equilibrio entre precisión y sensibilidad con el corte de 0.5. Va de 0 a 1.",
    "ECE": "Diferencia media entre la probabilidad que dice el modelo y la frecuencia real. Mientras más baja, más literal se puede leer la "
           "probabilidad.",
    "RMSE": "Error típico al predecir la nota, en puntos de nota. Mientras más bajo, mejor.",
    "calibración": "Un modelo está calibrado si, entre los vinos a los que da 30 %, cerca del 30 % son buenos.",
    "umbral": "Probabilidad mínima para marcar un vino como bueno. Con 0.5, se marca bueno todo vino con 50 % o más.",
    "sensibilidad": "De los vinos realmente buenos, qué porcentaje encuentra el modelo.",
    "precisión": "De los vinos que el modelo marca como buenos, qué porcentaje lo es.",
    "matriz de confusión": "Tabla de aciertos y errores: buenos encontrados, buenos perdidos, falsas alarmas y bien descartados.",
    "ROC": "Curva que muestra, para cada umbral, cuántos vinos buenos encuentra el modelo frente a cuántas falsas alarmas da.",
    "IC 95 %": "Intervalo de confianza: el rango donde razonablemente está el valor verdadero.",
    "bootstrap": "Repetir un cálculo muchas veces con muestras sorteadas de los mismos datos para medir cuánto puede variar.",
    "SHAP": "Reparte la predicción entre las variables: dice cuánto empujó cada una hacia «bueno» o hacia «no bueno».",
    "conjunto conforme": "Lista de respuestas compatibles con el vino, con una garantía estadística de acierto. Si trae las dos clases, el modelo "
                         "no tiene evidencia para decidir.",
    "p-valor": "Qué tan fácil sería ver una diferencia así solo por casualidad. Si es menor que 0.05, se acepta que la diferencia es real.",
    "Holm": "Ajuste de los p-valores para no declarar diferencias por azar cuando se hacen muchas comparaciones.",
    "Friedman": "Prueba global: ¿todos los modelos ocupan en promedio el mismo puesto en los 25 bloques?",
    "Nemenyi": "Compara los puestos de cada par de modelos después de Friedman. Es exigente: controla todas las comparaciones a la vez.",
    "Wilcoxon": "Compara las diferencias de AUC bloque a bloque usando sus rangos, sin suponer normalidad.",
    "t corregida": "Prueba t de Nadeau y Bengio: infla la varianza porque los 25 bloques comparten vinos. Es la más prudente.",
    "DeLong": "Compara dos AUC calculados sobre los mismos 5320 vinos.",
    "t bayesiana": "Versión bayesiana de la prueba t para bloques correlacionados: en lugar de un p-valor da la probabilidad de que cada modelo "
                   "sea mejor por más de 0.005 de AUC.",
    "delta de Cliff": "Tamaño del efecto: qué tan seguido un modelo supera al otro. Debajo de 0.147 se considera despreciable.",
    "d pareada": "Tamaño del efecto: la diferencia media dividida por su desviación estándar.",
    "pre-registrado": "Las reglas para decidir si el modelo gana se escribieron y se fecharon antes de correr los experimentos, para no "
                      "acomodarlas al resultado.",
    "ablación": "Quitar una pieza del modelo cada vez para medir cuánto aporta.",
    "robustez": "Qué tanto aguanta un modelo cuando cambian las condiciones: otra semilla, ruido en los datos, menos vinos u otro tipo de vino.",
    "núcleo": "Curva en forma de campana que decide cuánto pesa cada umbral según su distancia al corte de 7.",
    "fuera de bolsa": "Cada árbol deja por fuera cerca del 37 % de los vinos; esas predicciones son honestas porque ese árbol no los vio.",
}
GRUPOS_GLOSARIO = {
    "Lo básico": ["bueno", "probabilidad", "ordinal", "mediana", "desviación estándar", "correlación", "atípico", "prevalencia"],
    "Entrenar y examinar modelos": ["benchmark", "balanceo", "optimizador", "hiperparámetro", "validación anidada", "bloque", "pliegue",
                                    "fuera de muestra"],
    "Medir qué tan bueno es un modelo": ["AUC", "PR-AUC", "Brier", "F1", "ECE", "RMSE", "calibración", "umbral", "sensibilidad", "precisión",
                                         "matriz de confusión", "ROC", "IC 95 %", "bootstrap"],
    "Explicar y dudar": ["SHAP", "conjunto conforme"],
    "Decidir si una diferencia es real": ["p-valor", "Holm", "Friedman", "Nemenyi", "Wilcoxon", "t corregida", "DeLong", "t bayesiana",
                                          "delta de Cliff", "d pareada", "pre-registrado"],
    "El modelo original": ["ablación", "robustez", "núcleo", "fuera de bolsa"],
}


def unir(elementos):
    """Une una lista en español: «A», «A y B», «A, B y C»."""
    elementos = list(elementos)
    if len(elementos) <= 1:
        return "".join(elementos)
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


def minuscula(nombre):
    """Nombre de una variable para usarlo dentro de una frase; respeta siglas y símbolos como SO₂ o pH."""
    if len(nombre) > 1 and nombre[1].isupper():
        return nombre
    return nombre[0].lower() + nombre[1:]


def numero(valor, decimales=0):
    """Número con punto decimal y espacio fino de miles a partir de cinco cifras (norma de la RAE)."""
    texto = f"{abs(valor):.{decimales}f}"
    entero, _, fraccion = texto.partition(".")
    if len(entero) > 4:
        entero = f"{int(entero):,}".replace(",", " ")
    return ("−" if valor < 0 else "") + entero + (f".{fraccion}" if fraccion else "")


def p_valor(p):
    """p-valor legible: nunca en notación científica."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def termino(texto, clave=None):
    """Palabra técnica con su definición sencilla: al pasar el cursor, con el teclado o al tocarla (abre el glosario)."""
    clave = clave or texto
    definicion = GLOSARIO[clave]
    return html.Span(texto, className="termino", tabIndex=0, role="button", title=definicion,
                     **{"data-def": definicion, "data-clave": clave, "aria-label": f"{texto}: {definicion}"})


def encabezado(pregunta, apoyo=None, etiqueta=None, oscuro=False, ancla=None):
    """Título de sección escrito como la pregunta que responde."""
    hijos = [html.Div(etiqueta, className="etiqueta-seccion")] if etiqueta else []
    hijos.append(html.H2(pregunta))
    if apoyo:
        hijos.append(html.P(apoyo, className="apoyo"))
    args = {"id": ancla} if ancla else {}
    return html.Header(hijos, className="encabezado aparecer" + (" oscuro" if oscuro else ""), **args)


def idea(titular, *detalle):
    """Contenido de una lectura: la conclusión en una frase sencilla y, después, las cifras que la sostienen."""
    return [html.Strong(titular, className="titular")] + ([html.Span(list(detalle), className="detalle")] if detalle else [])


def lectura(contenido, id=None):
    """Frase de interpretación que acompaña a una figura."""
    args = {"id": id} if id else {}
    return html.Div([html.I(className="bi bi-lightbulb", **{"aria-hidden": "true"}), html.Div(contenido, **args)], className="lectura")


def como_leer(*texto):
    """Instrucción breve para leer una gráfica poco común."""
    return html.P([html.I(className="bi bi-eye", **{"aria-hidden": "true"}), html.Span([html.Strong("Cómo leerla: "), *texto])],
                  className="como-leer")


def tarjeta(titulo, cuerpo, pie=None, clase=""):
    hijos = ([html.H3(titulo, className="tarjeta-titulo")] if titulo else []) + (cuerpo if isinstance(cuerpo, list) else [cuerpo])
    if pie is not None:
        hijos.append(pie)
    return html.Section(hijos, className=f"tarjeta aparecer {clase}".strip())


def cifra(valor, etiqueta, decimales=0, sufijo="", i=0):
    """Cifra destacada que cuenta hasta su valor al aparecer."""
    texto = numero(valor, decimales) + sufijo
    return html.Div([html.Div(texto, className="cifra-valor", **{"data-contar": str(valor), "data-decimales": str(decimales), "data-sufijo": sufijo}),
                     html.Div(etiqueta, className="cifra-etiqueta")], className="cifra voltea", style={"--i": i})


def pendiente(que):
    """Aviso de un resultado que todavía no existe; nunca se rellena con cifras simuladas."""
    return html.Div([html.I(className="bi bi-hourglass-split", **{"aria-hidden": "true"}),
                     html.Div([html.Strong("Resultado en preparación. "),
                               f"{que} se publica aquí cuando termine el cómputo que falta; no se muestran cifras simuladas."])],
                    className="pendiente")


def fila(*columnas, anchos=None, clase="g-4"):
    """Fila de la rejilla de Bootstrap; ``anchos`` da el ancho en pantallas grandes de cada columna."""
    anchos = anchos or [12 // len(columnas)] * len(columnas)
    return dbc.Row([dbc.Col(c, lg=a, md=12) for c, a in zip(columnas, anchos)], className=clase)


def tabla(columnas, filas, destacar=None):
    """Tabla con encabezados que se explican solos: cada columna es (título, ayuda) y la ayuda aparece al pasar el cursor.

    ``destacar`` es el índice de la fila que se resalta (por ejemplo, la del modelo original).
    """
    cabecera = [html.Th(html.Abbr(t, title=a) if a else t) for t, a in columnas]
    cuerpo = [html.Tr([html.Td(c) for c in f], className="fila-destacada" if i == destacar else None) for i, f in enumerate(filas)]
    return dbc.Table([html.Thead(html.Tr(cabecera)), html.Tbody(cuerpo)], size="sm", hover=True, responsive=True, class_name="tabla-numeros mb-0")


def indice(entradas):
    """Enlaces a las secciones de una página larga; ``entradas`` es una lista de (ancla, texto)."""
    return html.Nav([html.Span("En esta página", className="indice-titulo")]
                    + [html.A(texto, href=f"#{ancla}", className="indice-enlace") for ancla, texto in entradas],
                    className="indice", **{"aria-label": "Secciones de esta página"})


def pasos(*textos):
    """Instrucciones numeradas y breves para usar una sección interactiva."""
    return html.Ol([html.Li([html.Span(str(i + 1), className="paso-num", **{"aria-hidden": "true"}), html.Span(t)]) for i, t in enumerate(textos)],
                   className="pasos")


def escala_notas(porcentajes, corte=7):
    """Las notas de la cata como baldosas, con el corte que separa a los vinos buenos.

    ``porcentajes`` es un dict {nota: porcentaje de vinos}; muestra por qué la nota es ordinal.
    """
    tope = max(porcentajes.values()) or 1
    baldosas = []
    for nota, pct in sorted(porcentajes.items()):
        if nota == corte:
            baldosas.append(html.Div(html.Span(f"corte: {corte}"), className="escala-corte", **{"aria-hidden": "true"}))
        clase = "bueno" if nota >= corte else "malo"
        baldosas.append(html.Div([html.Div(html.Div(style={"height": f"{max(4, 100 * pct / tope):.0f}%"}), className="escala-barra"),
                                  html.Div(str(nota), className="escala-nota"), html.Div(f"{numero(pct, 1 if pct < 10 else 0)} %", className="escala-pct")],
                                 className=f"escala-baldosa {clase}", title=f"Nota {nota}: {numero(pct, 1)} % de los vinos"))
    buenos = sum(p for n, p in porcentajes.items() if n >= corte)
    return html.Figure([
        html.Div(baldosas, className="escala", role="img",
                 **{"aria-label": f"Notas de {min(porcentajes)} a {max(porcentajes)}. Son buenos los vinos con {corte} o más: {numero(buenos)} %."}),
        html.Div([html.Span([html.I(className="escala-muestra malo"), f"No bueno: nota de {corte - 1} o menos"]),
                  html.Span([html.I(className="escala-muestra bueno"), f"Bueno: nota de {corte} o más ({numero(buenos)} % de los vinos)"])],
                 className="leyenda mt-2"),
    ], className="escala-figura")


def recorrido(actual):
    """Pasos del tablero como una ruta; marca la página actual y destaca la siguiente."""
    rutas = [p[0] for p in PAGINAS]
    i = rutas.index(actual)
    eslabones = []
    for k, (ruta, nombre, icono, que) in enumerate(PAGINAS):
        estado = "hecho" if k < i else "actual" if k == i else "siguiente" if k == i + 1 else "futuro"
        cuerpo = [html.Span(str(k + 1), className="eslabon-num"), html.Span([html.Strong(nombre), html.Small(que)], className="eslabon-texto")]
        eslabones.append(html.Li(dcc.Link(cuerpo, href=ruta, className=f"eslabon {estado}"), **({"aria-current": "step"} if k == i else {})))
    return html.Ol(eslabones, className="recorrido")


def siguiente(actual, texto=None):
    """Cierre de página: dónde está el visitante dentro del recorrido y a dónde conviene ir después."""
    rutas = [p[0] for p in PAGINAS]
    i = rutas.index(actual)
    if i + 1 < len(PAGINAS):
        ruta, nombre, _, que = PAGINAS[i + 1]
        boton = dcc.Link([f"Siguiente: {nombre}", html.I(className="bi bi-arrow-right")], href=ruta, className="boton")
        frase = texto or f"{que}."
    else:
        boton = dcc.Link([html.I(className="bi bi-arrow-counterclockwise"), "Volver al inicio"], href="/", className="boton")
        frase = texto or "Este es el final del recorrido."
    return html.Section(dbc.Container([
        html.Div([html.Div([html.Div("El recorrido", className="etiqueta-seccion"), html.P(frase, className="siguiente-frase")]), boton],
                 className="siguiente-cabecera"),
        recorrido(actual),
    ], fluid="xl"), className="banda banda-siguiente", **{"aria-label": "Recorrido del tablero"})


def panel_glosario():
    """Contenido del panel lateral: guía rápida para leer el tablero y glosario completo, con buscador."""
    guia = html.Ul([
        html.Li([html.Span(className="guia-muestra", style={"background": "#1d5fc4"}), html.Span([html.Strong("Azul"), ": vinos buenos y el modelo original."])]),
        html.Li([html.Span(className="guia-muestra", style={"background": "#d07a10"}), html.Span([html.Strong("Ocre"), ": vinos no buenos y el rival."])]),
        html.Li([html.I(className="bi bi-lightbulb guia-icono"), html.Span("Cada gráfica trae un recuadro azul claro: la primera frase, en negrita, es la conclusión.")]),
        html.Li([html.I(className="bi bi-eye guia-icono"), html.Span("«Cómo leerla» explica las gráficas poco comunes.")]),
        html.Li([html.Span("AUC", className="termino guia-termino"), html.Span("Las palabras con subrayado punteado tienen definición: pasa el cursor o tócalas.")]),
    ], className="guia-rapida")
    grupos = []
    for grupo, claves in GRUPOS_GLOSARIO.items():
        grupos.append(html.Div([html.H3(grupo, className="glosario-grupo")]
                               + [html.Div([html.Dt(c), html.Dd(GLOSARIO[c])], className="glosario-entrada", **{"data-termino": c}) for c in claves],
                               className="glosario-bloque"))
    return [html.H3("Cómo leer este tablero", className="glosario-grupo"), guia,
            html.Label("Buscar en el glosario", htmlFor="glosario-buscar", className="visually-hidden"),
            dbc.Input(id="glosario-buscar", type="search", placeholder="Buscar una palabra…", class_name="mb-3"),
            html.Dl(grupos, id="glosario-lista", className="glosario"),
            html.P("No hay palabras que coincidan.", id="glosario-vacio", className="text-secondary small", hidden=True)]
