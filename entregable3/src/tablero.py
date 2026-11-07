from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

VERSION = "1.0 · noviembre de 2026"
AUTORES = "Roger Daza, Matteo Del Sole y Rafael Romero"
CORTOS = {"Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM", "KNN": "KNN", "Regresión Logística (L1/L2)": "RL",
          "Decision Tree": "Árbol", "Naive Bayes": "NB", "LightGBM": "LGBM", "CatBoost": "CB", "Bosque Ordinal Focalizado": "BOF"}
GLOSARIO = {
    "AUC": "Si tomas al azar un vino bueno y uno que no lo es, es la probabilidad de que el modelo le dé más puntaje al bueno. 0.5 es azar y 1 es perfecto.",
    "PR-AUC": "Mide qué tan limpia es la lista de vinos que el modelo marca como buenos. Es útil porque los vinos buenos son minoría.",
    "Brier": "Error cuadrático de las probabilidades: compara lo que dijo el modelo con lo que pasó. Mientras más bajo, mejor.",
    "F1": "Equilibrio entre precisión y sensibilidad con el corte de 0.5.",
    "ECE": "Diferencia media entre la probabilidad que dice el modelo y la frecuencia real. Mientras más baja, más literal se puede leer la probabilidad.",
    "SHAP": "Reparte la predicción entre las variables: dice cuánto empujó cada una hacia «bueno» o hacia «no bueno».",
    "conjunto conforme": "Lista de respuestas compatibles con el vino, con una garantía estadística de acierto. Si trae las dos clases, el modelo no tiene evidencia para decidir.",
    "validación anidada": "Dos bucles: el interno elige los hiperparámetros y el externo mide el desempeño con vinos que el modelo nunca vio.",
    "bloque": "Una partición de entrenamiento y prueba. Hay 25: 5 repeticiones de 5 pliegues.",
    "Friedman": "Prueba global: ¿todos los modelos ocupan en promedio el mismo puesto en los 25 bloques?",
    "Nemenyi": "Compara los rangos de cada par de modelos después de Friedman. Es exigente: controla todas las comparaciones a la vez.",
    "Wilcoxon": "Compara las diferencias de AUC bloque a bloque usando sus rangos, sin suponer normalidad.",
    "t corregida": "Prueba t de Nadeau y Bengio: infla la varianza porque los 25 bloques comparten vinos. Es la más prudente.",
    "DeLong": "Compara dos AUC calculados sobre los mismos 5320 vinos.",
    "t bayesiana": "Versión bayesiana de la prueba t para bloques correlacionados: en lugar de un p-valor da la probabilidad de que cada modelo sea mejor por más de 0.005 de AUC.",
    "Holm": "Ajuste de los p-valores para no declarar diferencias por azar cuando se hacen muchas comparaciones.",
    "delta de Cliff": "Tamaño del efecto: qué tan seguido un modelo supera al otro. Debajo de 0.147 se considera despreciable.",
    "d pareada": "Tamaño del efecto: la diferencia media dividida por su desviación estándar.",
    "calibración": "Un modelo está calibrado si, entre los vinos a los que da 30 %, cerca del 30 % son buenos.",
    "ablación": "Quitar una pieza del modelo cada vez para medir cuánto aporta.",
    "núcleo": "Curva en forma de campana que decide cuánto pesa cada umbral según su distancia al corte de 7.",
    "fuera de bolsa": "Cada árbol deja por fuera cerca del 37 % de los vinos; esas predicciones son honestas porque ese árbol no los vio.",
    "sensibilidad": "De los vinos realmente buenos, qué porcentaje encuentra el modelo.",
    "precisión": "De los vinos que el modelo marca como buenos, qué porcentaje lo es.",
    "umbral": "Probabilidad mínima para marcar un vino como bueno.",
}


def unir(elementos):
    """Une una lista en español: «A», «A y B», «A, B y C»."""
    elementos = list(elementos)
    if len(elementos) <= 1:
        return "".join(elementos)
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


def termino(texto, clave=None):
    """Palabra técnica con su definición sencilla al pasar el cursor o al enfocarla con el teclado."""
    definicion = GLOSARIO[clave or texto]
    return html.Span(texto, className="termino", tabIndex=0, title=definicion, **{"data-def": definicion, "aria-label": f"{texto}: {definicion}"})


def encabezado(pregunta, apoyo=None, etiqueta=None, oscuro=False):
    """Título de sección escrito como la pregunta que responde."""
    hijos = [html.Div(etiqueta, className="etiqueta-seccion")] if etiqueta else []
    hijos.append(html.H2(pregunta))
    if apoyo:
        hijos.append(html.P(apoyo, className="apoyo"))
    return html.Header(hijos, className="encabezado aparecer" + (" oscuro" if oscuro else ""))


def lectura(contenido, id=None):
    """Frase de interpretación que acompaña a una figura."""
    args = {"id": id} if id else {}
    return html.Div([html.I(className="bi bi-lightbulb", **{"aria-hidden": "true"}), html.Div(contenido, **args)], className="lectura")


def tarjeta(titulo, cuerpo, pie=None, clase=""):
    hijos = ([html.H3(titulo, className="tarjeta-titulo")] if titulo else []) + (cuerpo if isinstance(cuerpo, list) else [cuerpo])
    if pie is not None:
        hijos.append(pie)
    return html.Section(hijos, className=f"tarjeta aparecer {clase}".strip())


def numero(valor, decimales=0):
    """Número con punto decimal y espacio fino de miles a partir de cinco cifras (norma de la RAE)."""
    texto = f"{abs(valor):.{decimales}f}"
    entero, _, fraccion = texto.partition(".")
    if len(entero) > 4:
        entero = f"{int(entero):,}".replace(",", " ")
    return ("−" if valor < 0 else "") + entero + (f".{fraccion}" if fraccion else "")


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
