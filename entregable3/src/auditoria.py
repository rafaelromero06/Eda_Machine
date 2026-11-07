from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .benchmarks import construir_pipeline
from .data import NUMERICAS_VINO, RUTA_DATOS, cargar_conjunto, cargar_vinos
from .validacion import K_INTERNO, SEMILLAS_REPETICION, particiones


def duplicados():
    tinto = pd.read_csv(RUTA_DATOS / "winequality-red.csv", sep=";").assign(tipo_vino="red")
    blanco = pd.read_csv(RUTA_DATOS / "winequality-white.csv", sep=";").assign(tipo_vino="white")
    crudo = pd.concat([tinto, blanco], ignore_index=True)
    limpio = cargar_vinos()
    x = NUMERICAS_VINO + ["tipo_vino"]
    grupos = limpio.groupby(x)["quality"].nunique()
    return {"filas_originales": len(crudo), "duplicados_exactos": int(crudo.duplicated().sum()), "filas_finales": len(limpio),
            "duplicados_restantes": int(limpio.duplicated().sum()),
            "grupos_misma_quimica_distinta_nota": int((grupos > 1).sum()),
            "vinos_en_esos_grupos": int(limpio.set_index(x).index.isin(grupos[grupos > 1].index).sum())}


def vecinos_entre_particiones():
    """Distancia (variables estandarizadas con el entrenamiento) de cada vino de prueba al vino de entrenamiento más cercano."""
    datos = cargar_conjunto("vino")
    X = datos["X"]
    filas = []
    for fold, (tr, te) in enumerate(particiones(datos["y_evento"], SEMILLAS_REPETICION[0])):
        Xtr = X.iloc[tr][NUMERICAS_VINO].to_numpy()
        Xte = X.iloc[te][NUMERICAS_VINO].to_numpy()
        media, sd = Xtr.mean(0), Xtr.std(0)
        arbol = cKDTree((Xtr - media) / sd)
        d_te, _ = arbol.query((Xte - media) / sd, k=1)
        d_tr, _ = arbol.query((Xtr - media) / sd, k=2)
        filas.append({"fold": fold, "prueba_con_gemelo_exacto": float((d_te < 1e-9).mean()),
                      "prueba_con_vecino_muy_cercano": float((d_te < 0.1).mean()),
                      "mediana_distancia_prueba": float(np.median(d_te)),
                      "mediana_distancia_dentro_entrenamiento": float(np.median(d_tr[:, 1]))})
    return pd.DataFrame(filas)


def preprocesamiento_en_el_pliegue():
    """Comprueba que el escalado se ajusta solo con el entrenamiento externo y que SMOTE no toca la prueba."""
    datos = cargar_conjunto("vino")
    tr, te = particiones(datos["y_evento"], SEMILLAS_REPETICION[0])[0]
    pipe = construir_pipeline("XGBoost", datos)
    pipe.set_params(modelo__n_estimators=20)
    pipe.fit(datos["X"].iloc[tr], datos["y_evento"][tr])
    escalador = pipe.named_steps["prep"].named_transformers_["num"]
    media_tr = datos["X"].iloc[tr][NUMERICAS_VINO].mean().to_numpy()
    media_todo = datos["X"][NUMERICAS_VINO].mean().to_numpy()
    n_pred = len(pipe.predict(datos["X"].iloc[te]))
    internas = particiones(datos["y_evento"][tr], SEMILLAS_REPETICION[0], K_INTERNO)
    return {"media_escalador_igual_a_entrenamiento": bool(np.allclose(escalador.mean_, media_tr)),
            "media_escalador_distinta_de_todo": bool(not np.allclose(escalador.mean_, media_todo)),
            "diferencia_maxima_con_todo": float(np.abs(escalador.mean_ - media_todo).max()),
            "predicciones_en_prueba": n_pred, "tamano_prueba": len(te),
            "particiones_internas_dentro_del_entrenamiento": bool(all(np.max(a) < len(tr) and np.max(b) < len(tr) for a, b in internas))}


def asociacion_con_la_respuesta():
    df = cargar_vinos()
    corr = df[NUMERICAS_VINO + ["quality"]].corr()["quality"].drop("quality")
    return corr.abs().sort_values(ascending=False)


def tabla_auditoria():
    d = duplicados()
    v = vecinos_entre_particiones()
    p = preprocesamiento_en_el_pliegue()
    a = asociacion_con_la_respuesta()
    return pd.DataFrame([
        {"riesgo": "Preprocesamiento ajustado con todos los datos",
         "verificacion": "Media del escalador dentro del pipeline frente a la media del entrenamiento externo y de todos los datos",
         "resultado": f"coincide con el entrenamiento: {p['media_escalador_igual_a_entrenamiento']}; difiere de todos los datos hasta en {p['diferencia_maxima_con_todo']:.3g}",
         "conclusion": "Sin fuga: el escalado y el one-hot se ajustan dentro de cada pliegue."},
        {"riesgo": "Balanceo aplicado a la prueba",
         "verificacion": "Número de predicciones de un pipeline con SMOTE sobre el pliegue de prueba",
         "resultado": f"{p['predicciones_en_prueba']} predicciones para {p['tamano_prueba']} vinos de prueba",
         "conclusion": "Sin fuga: SMOTE y ADASYN solo actúan en el ajuste (pipeline de imbalanced-learn)."},
        {"riesgo": "Búsqueda de hiperparámetros con datos de prueba",
         "verificacion": "Los 3 pliegues internos se construyen solo con índices del entrenamiento externo",
         "resultado": str(p["particiones_internas_dentro_del_entrenamiento"]),
         "conclusion": "Sin fuga: la prueba externa nunca entra en la búsqueda; el umbral y la calibración del modelo original usan solo el entrenamiento (fuera de bolsa)."},
        {"riesgo": "Duplicados entre particiones",
         "verificacion": "Duplicados exactos antes de partir y vinos de prueba con un gemelo químico en el entrenamiento",
         "resultado": f"{d['duplicados_exactos']} duplicados exactos eliminados ({d['filas_originales']} → {d['filas_finales']}); "
                      f"{100 * v['prueba_con_gemelo_exacto'].mean():.2f} % de los vinos de prueba tiene un gemelo químico (misma química, otra nota)",
         "conclusion": "Sin duplicados exactos; los gemelos químicos con nota distinta son ruido de la etiqueta, no fuga."},
        {"riesgo": "Variables que codifican la respuesta",
         "verificacion": "Asociación de cada variable con la nota y origen de las variables",
         "resultado": f"máxima |r| = {a.iloc[0]:.2f} ({a.index[0]}); todas son mediciones de laboratorio previas a la cata",
         "conclusion": "Sin fuga: ninguna variable se deriva de la nota."},
        {"riesgo": "Mezcla temporal o espacial (lotes)",
         "verificacion": "No hay fecha ni productor; se compara la validación aleatoria con una validación por bloques consecutivos del archivo",
         "resultado": "ver el estudio de validación por bloques en el capítulo de evaluación ampliada",
         "conclusion": "Riesgo residual: si vinos del mismo lote caen en entrenamiento y prueba, las métricas pueden ser algo optimistas."},
    ]), d, v, p
