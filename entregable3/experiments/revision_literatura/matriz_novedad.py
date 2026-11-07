import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
BUSQUEDA = json.loads((AQUI / "busqueda_2026-10-26.json").read_text(encoding="utf-8"))

JUICIO = {
    ("C1", "OpenAlex"): (50, 5, "Bosques ordinales con puntajes (Janitza et al., 2016) y sin puntajes (Tutz, 2022), comparaciones de ensambles ordinales (Buczak, 2024) y RF/SVM multiclase para vino. Ninguno usa los niveles para un evento binario."),
    ("C1", "Crossref"): (25, 4, "Aplicaciones de RF a vino que tratan la nota como clase o la binarizan sin usar los niveles; bosque ordinal por conjuntos aproximados (Wang et al., 2015)."),
    ("C2", "OpenAlex"): (50, 4, "Extra-Trees ya se usó para vino como clasificador estándar (binario o multiclase): la novedad no puede ser el algoritmo base."),
    ("C2", "Semantic Scholar"): (4, 3, "Los mismos estudios de aplicación a vino; sin información ordinal."),
    ("C2", "Crossref"): (25, 1, "Comparación de ensambles de árboles en vino sin fuga (Chen, 2025), con la nota como clase multiclase."),
    ("C3", "OpenAlex"): (50, 4, "XGBOrdinal (Frank y Hall con XGBoost), Classifier Pooling (clasificación ordinal agnóstica), pérdidas dependientes de la etiqueta y regresión logística para la preferencia de vino. Su objetivo es predecir la clase ordinal, no un evento binario."),
    ("C3", "arXiv"): (3, 1, "OGBoost: boosting ordinal con variable latente; predice la clase ordinal."),
    ("C4", "OpenAlex"): (25, 1, "El CRPS ponderado por umbral se usa para evaluar o entrenar pronósticos (posprocesamiento de viento extremo), no como criterio de partición de árboles."),
    ("C5", "OpenAlex"): (50, 5, "Ordered Forest (Lechner y Okasa) estima cada probabilidad acumulada con un bosque separado y las resta para dar la clase; bosques ordinales con efectos mixtos, fabOF y bosques de Wilcoxon. Ninguno combina umbrales vecinos para un evento binario."),
    ("C5", "Crossref"): (25, 4, "Bosques ordinales (Wang et al., 2015; Janitza et al., 2016), RF para clasificación ordinal monótona (2019) y MEfabOF (2024)."),
    ("C6", "OpenAlex"): (30, 1, "Un estudio en español de IA para la calidad del vino, con clasificadores estándar."),
    ("C7", "Crossref"): (25, 3, "fabOF, MEfabOF y comparaciones de bosques ordinales; la probabilidad de excedencia no aparece como objetivo de un bosque ordinal."),
    ("C8", "OpenAlex"): (50, 2, "Aprovechar etiquetas ordinales para predicciones continuas o suavizadas, siempre con redes neuronales."),
    ("C8", "Crossref"): (25, 3, "Pérdida de información al binarizar un desenlace ordinal (Harrell, 2018), reducción de la regresión ordinal a clasificación binaria extendida (Li y Lin, 2007) y etiquetas suaves ordinales con redes."),
}


def matriz():
    filas = []
    for c in BUSQUEDA["consultas"]:
        total = c["total"] or 0
        revisados, relevantes, obs = JUICIO.get((c["cadena"], c["base"]), (min(total, len(c["items"])), 0, "Sin trabajos relevantes."))
        if c["base"] == "Crossref" and (c["cadena"], c["base"]) not in JUICIO:
            revisados = 25
        filas.append({"cadena": c["cadena"], "base": c["base"], "consulta": c["consulta"], "resultados": total,
                      "revisados": revisados, "relevantes": relevantes, "observacion": obs})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    m = matriz()
    m.to_csv(AQUI / "matriz_novedad.csv", index=False)
    print(m.drop(columns="consulta").to_string(index=False))
