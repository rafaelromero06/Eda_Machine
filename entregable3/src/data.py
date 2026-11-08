import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_BASE = Path(__file__).resolve().parents[1]
RUTA_DATOS = RUTA_BASE / "data"
RUTA_ADICIONALES = RUTA_DATOS / "adicionales"
UMBRAL_VINO = 7

NUMERICAS_VINO = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]
CATEGORICAS_VINO = ["tipo_vino"]

DICCIONARIO_VINO = {
    "fixed acidity": ("Acidez fija", "g/L de ácido tartárico", "Ácidos no volátiles; aportan frescura."),
    "volatile acidity": ("Acidez volátil", "g/L de ácido acético", "En exceso da olor a vinagre."),
    "citric acid": ("Ácido cítrico", "g/L", "Aporta frescura y sabor."),
    "residual sugar": ("Azúcar residual", "g/L", "Azúcar que queda tras la fermentación."),
    "chlorides": ("Cloruros", "g/L de cloruro de sodio", "Contenido de sal."),
    "free sulfur dioxide": ("SO₂ libre", "mg/L", "Protege el vino de la oxidación y de microbios."),
    "total sulfur dioxide": ("SO₂ total", "mg/L", "SO₂ libre más el combinado."),
    "density": ("Densidad", "g/cm³", "Depende sobre todo del alcohol y del azúcar."),
    "pH": ("pH", "escala 0–14", "Acidez global; los vinos están entre 2.7 y 4."),
    "sulphates": ("Sulfatos", "g/L de sulfato de potasio", "Aditivo que contribuye al SO₂."),
    "alcohol": ("Alcohol", "% vol.", "Grado alcohólico."),
    "tipo_vino": ("Tipo de vino", "tinto / blanco", "Vinho Verde tinto o blanco."),
    "quality": ("Calidad", "nota de 0 a 10", "Mediana de al menos tres catadores."),
}

ADICIONALES = {
    "abalone": {
        "archivo": "abalone.csv", "respuesta": "Rings", "umbral": 11, "excluir": [],
        "categoricas": ["Sex"], "titulo": "Abalone (edad por anillos)",
        "evento": "abulón de 11 anillos o más (≈ 12.5 años o más)",
    },
    "cardiotocografia": {
        "archivo": "cardiotocography.csv", "respuesta": "NSP", "umbral": 2, "excluir": ["CLASS"],
        "categoricas": [], "titulo": "Cardiotocografía (estado fetal)",
        "evento": "registro sospechoso o patológico (NSP 2 o 3)",
    },
    "estudiantes": {
        "archivo": "student_performance.csv", "respuesta": "G3", "umbral": 10, "excluir": ["G1", "G2"],
        "categoricas": None, "titulo": "Student Performance (portugués)",
        "evento": "estudiante aprobado (nota final G3 de 10 o más sobre 20)",
    },
    "corazon": {
        "archivo": "heart_disease.csv", "respuesta": "num", "umbral": 1, "excluir": [],
        "categoricas": ["cp", "restecg", "slope", "thal"], "titulo": "Heart Disease (Cleveland)",
        "evento": "presencia de enfermedad coronaria (num de 1 a 4)",
    },
}


@lru_cache(maxsize=1)
def cargar_vinos() -> pd.DataFrame:
    """Une los dos CSV de Vinho Verde, agrega ``tipo_vino`` y quita duplicados exactos (5320 filas)."""
    tinto = pd.read_csv(RUTA_DATOS / "winequality-red.csv", sep=";")
    blanco = pd.read_csv(RUTA_DATOS / "winequality-white.csv", sep=";")
    tinto["tipo_vino"] = "red"
    blanco["tipo_vino"] = "white"
    df = pd.concat([tinto, blanco], ignore_index=True)
    return df.drop_duplicates().reset_index(drop=True)


def separar_vinos(df: pd.DataFrame | None = None):
    """Devuelve (X, y_ordinal, y_evento) con el mismo orden de filas del Entregable 2."""
    df = cargar_vinos() if df is None else df
    X = df[NUMERICAS_VINO + CATEGORICAS_VINO].copy()
    y_ord = df["quality"].astype(int).to_numpy()
    return X, y_ord, (y_ord >= UMBRAL_VINO).astype(int)


def esquema_vino() -> dict:
    """Nombre, tipo, unidad y rango válido de cada variable de entrada (para metadata.json y el formulario)."""
    df = cargar_vinos()
    esquema = {}
    for c in NUMERICAS_VINO:
        nombre, unidad, desc = DICCIONARIO_VINO[c]
        lo, hi = float(df[c].min()), float(df[c].max())
        margen = 0.1 * (hi - lo)
        esquema[c] = {"etiqueta": nombre, "tipo": "float", "unidad": unidad, "descripcion": desc,
                      "minimo_observado": lo, "maximo_observado": hi,
                      "minimo_valido": round(max(0.0, lo - margen), 4), "maximo_valido": round(hi + margen, 4),
                      "mediana": float(df[c].median())}
    nombre, unidad, desc = DICCIONARIO_VINO["tipo_vino"]
    esquema["tipo_vino"] = {"etiqueta": nombre, "tipo": "categorical", "unidad": unidad, "descripcion": desc,
                            "categorias": ["red", "white"]}
    return esquema


@lru_cache(maxsize=None)
def cargar_conjunto(nombre: str) -> dict:
    """Carga un conjunto (``vino`` o uno de ADICIONALES) en un formato común.

    Returns
    -------
    dict con X (DataFrame), y_ord, y_evento, umbral, numericas, categoricas y titulo.
    """
    if nombre == "vino":
        X, y_ord, y_ev = separar_vinos()
        return {"X": X, "y_ord": y_ord, "y_evento": y_ev, "umbral": UMBRAL_VINO,
                "numericas": list(NUMERICAS_VINO), "categoricas": list(CATEGORICAS_VINO),
                "titulo": "Wine Quality (Vinho Verde)", "imputar": False}
    spec = ADICIONALES[nombre]
    df = pd.read_csv(RUTA_ADICIONALES / spec["archivo"]).drop_duplicates().reset_index(drop=True)
    y_ord = df[spec["respuesta"]].astype(int).to_numpy()
    X = df.drop(columns=[spec["respuesta"]] + spec["excluir"])
    categoricas = spec["categoricas"]
    if categoricas is None:
        categoricas = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in categoricas:
        X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
    numericas = [c for c in X.columns if c not in categoricas]
    return {"X": X, "y_ord": y_ord, "y_evento": (y_ord >= spec["umbral"]).astype(int), "umbral": spec["umbral"],
            "numericas": numericas, "categoricas": list(categoricas), "titulo": spec["titulo"], "imputar": True}


def resumen_conjuntos(nombres=("vino", "abalone", "cardiotocografia", "estudiantes", "corazon")) -> pd.DataFrame:
    """Tamaño, dimensión, niveles de la respuesta y prevalencia del evento de cada conjunto."""
    filas = []
    for n in nombres:
        c = cargar_conjunto(n)
        filas.append({"conjunto": n, "titulo": c["titulo"], "n": len(c["y_ord"]), "p_numericas": len(c["numericas"]),
                      "p_categoricas": len(c["categoricas"]), "niveles": int(np.unique(c["y_ord"]).size),
                      "umbral": c["umbral"], "prevalencia": float(c["y_evento"].mean()),
                      "faltantes": int(c["X"].isna().sum().sum())})
    return pd.DataFrame(filas)


RUTA_APP = RUTA_DATOS / "app"
RUTA_ARTEFACTOS = RUTA_BASE / "artifacts"


@lru_cache(maxsize=None)
def tabla_app(nombre: str):
    """Tabla precalculada para el dashboard (None si todavía no existe)."""
    ruta = RUTA_APP / nombre
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


@lru_cache(maxsize=None)
def json_app(nombre: str):
    import json
    ruta = RUTA_APP / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


_MODELO_EN_MEMORIA: dict = {}
_CANDADO_MODELO = threading.Lock()


def modelo_final(carpeta: str):
    """Pipeline final serializado y su metadata, con un solo modelo en memoria a la vez (límite de Render).

    El modelo anterior se libera antes de cargar el nuevo, así el pico de memoria es el de un solo modelo.
    """
    import gc
    import json
    import joblib
    with _CANDADO_MODELO:
        if carpeta not in _MODELO_EN_MEMORIA:
            _MODELO_EN_MEMORIA.clear()
            gc.collect()
            ruta = RUTA_ARTEFACTOS / carpeta
            _MODELO_EN_MEMORIA[carpeta] = (joblib.load(ruta / "modelo.joblib"),
                                           json.loads((ruta / "metadata.json").read_text(encoding="utf-8")))
        return _MODELO_EN_MEMORIA[carpeta]


@lru_cache(maxsize=None)
def metadata_final(carpeta: str):
    import json
    ruta = RUTA_ARTEFACTOS / carpeta / "metadata.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
