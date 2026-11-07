from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import (CATEGORICAS_VINO, DICCIONARIO_VINO, NUMERICAS_VINO, UMBRAL_VINO, cargar_vinos, esquema_vino, json_app,
                   metadata_final, modelo_final)

ORIGINAL = "Bosque Ordinal Focalizado"
CARPETAS = {ORIGINAL: "bosque_ordinal_focalizado", "Random Forest": "random_forest", "XGBoost": "xgboost", "SVM": "svm",
            "KNN": "knn", "Regresión Logística (L1/L2)": "regresion_logistica_l1_l2", "Decision Tree": "decision_tree",
            "Naive Bayes": "naive_bayes", "LightGBM": "lightgbm", "CatBoost": "catboost"}
COLUMNAS = NUMERICAS_VINO + CATEGORICAS_VINO
ESQUEMA = esquema_vino()
TIPOS = {"tinto": "red", "red": "red", "blanco": "white", "white": "white"}
LIMITE_LOTE = 5000
MEJORA_MINIMA = 0.005


def disponibles():
    """Modelos con artefacto serializado, el modelo original primero."""
    return [m for m in CARPETAS if metadata_final(CARPETAS[m])]


def es_provisional(modelo):
    meta = metadata_final(CARPETAS[modelo]) or {}
    return bool(meta.get("metricas_bucle_externo", {}).get("simulado", False))


def validar(valores):
    """Mensajes de error de los valores numéricos, en el orden de ``NUMERICAS_VINO``."""
    errores = []
    for c, v in zip(NUMERICAS_VINO, valores):
        e = ESQUEMA[c]
        if v is None or v == "":
            errores.append(f"{e['etiqueta']}: falta el valor.")
            continue
        try:
            numero = float(v)
        except (TypeError, ValueError):
            errores.append(f"{e['etiqueta']}: «{v}» no es un número.")
            continue
        if not e["minimo_valido"] <= numero <= e["maximo_valido"]:
            errores.append(f"{e['etiqueta']}: {v} está fuera del rango válido "
                           f"({e['minimo_valido']:g} a {e['maximo_valido']:g} {e['unidad']}).")
    return errores


def armar_vino(valores, tipo):
    return pd.DataFrame([dict(zip(COLUMNAS, [float(v) for v in valores] + [tipo]))])


def _probabilidad(pipe, X):
    modelo = pipe.named_steps["modelo"]
    if hasattr(modelo, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-pipe.decision_function(X)))


def evaluar(modelo, X):
    """Probabilidad, clase y conjunto conforme al 90 % de un vino."""
    pipe, _ = modelo_final(CARPETAS[modelo])
    p = float(_probabilidad(pipe, X)[0])
    q = (json_app("umbrales_conformes.json") or {}).get(modelo, {})
    umbral = q.get("0.1")
    conjunto = [] if umbral is None else [c for c, no_conforme in (("bueno", 1 - p), ("no bueno", p)) if no_conforme <= umbral]
    return {"modelo": modelo, "p": p, "clase": "bueno" if p >= 0.5 else "no bueno", "conjunto": conjunto,
            "probabilistico": hasattr(pipe.named_steps["modelo"], "predict_proba"), "cobertura": q.get("cobertura_0.1"),
            "provisional": es_provisional(modelo)}


def explicar(modelo, X, fondo):
    """Aportes SHAP ordenados por magnitud, valor base y escala de la explicación."""
    from .explicaciones import Explicador

    pipe, _ = modelo_final(CARPETAS[modelo])
    exp = Explicador(modelo, pipe, fondo)
    aportes, base = exp.explicar(X)
    orden = aportes.abs().sort_values(ascending=False).index
    return {"aportes": [{"variable": c, "etiqueta": DICCIONARIO_VINO[c][0], "valor": float(aportes[c])} for c in orden],
            "base": base, "escala": exp.escala}


@lru_cache(maxsize=2)
def _vinos_de_tipo(tipo):
    d = cargar_vinos()
    d = d[d["tipo_vino"] == tipo]
    X = d[NUMERICAS_VINO].to_numpy(dtype=float)
    media, escala = X.mean(axis=0), X.std(axis=0)
    return (X - media) / escala, d["quality"].to_numpy(), media, escala


def gemelos(valores, tipo, k=5):
    """Notas reales de los ``k`` vinos más parecidos del mismo tipo, sin contar copias exactas."""
    Z, notas, media, escala = _vinos_de_tipo(tipo)
    z = (np.asarray(valores, dtype=float) - media) / escala
    distancia = np.sqrt(((Z - z) ** 2).sum(axis=1))
    orden = np.argsort(distancia)
    orden = orden[distancia[orden] > 1e-9][:k]
    return {"notas": [int(n) for n in notas[orden]], "buenos": int((notas[orden] >= UMBRAL_VINO).sum()),
            "distancia_media": float(distancia[orden].mean())}


def palancas(modelo, X, n=3):
    """Cambios de una sola variable (dentro del 90 % central de los vinos) que más suben la probabilidad del modelo."""
    pipe, _ = modelo_final(CARPETAS[modelo])
    base = float(_probabilidad(pipe, X)[0])
    vinos = cargar_vinos()
    cuantiles = np.linspace(0.05, 0.95, 9)
    copias = []
    for c in NUMERICAS_VINO:
        rejilla = pd.concat([X] * cuantiles.size, ignore_index=True)
        rejilla[c] = vinos[c].quantile(cuantiles).to_numpy()
        copias.append(rejilla)
    p = _probabilidad(pipe, pd.concat(copias, ignore_index=True)).reshape(len(NUMERICAS_VINO), cuantiles.size)
    salida = []
    for i, c in enumerate(NUMERICAS_VINO):
        j = int(np.argmax(p[i]))
        if p[i, j] - base >= MEJORA_MINIMA:
            salida.append({"variable": c, "etiqueta": DICCIONARIO_VINO[c][0], "unidad": DICCIONARIO_VINO[c][1],
                           "desde": float(X[c].iloc[0]), "hasta": float(copias[i][c].iloc[j]), "p_desde": base, "p_hasta": float(p[i, j])})
    return sorted(salida, key=lambda f: -f["p_hasta"])[:n]


def distribucion_nota(modelo, X):
    """Niveles de la nota y su probabilidad; solo el modelo original predice la nota completa."""
    if modelo != ORIGINAL:
        return None
    pipe, _ = modelo_final(CARPETAS[modelo])
    bof = pipe.named_steps["modelo"]
    _, prob = bof.predict_niveles(pipe.named_steps["prep"].transform(X))
    return np.concatenate([[bof.umbrales_[0] - 1], bof.umbrales_]).astype(float), prob[0]


def evaluar_lote(modelo, tabla):
    """Predice un lote de vinos. Devuelve (tabla ordenada por probabilidad o None, lista de errores)."""
    faltan = [c for c in COLUMNAS if c not in tabla.columns]
    if faltan:
        return None, ["Faltan columnas en el archivo: " + ", ".join(faltan) + "."]
    if tabla.empty:
        return None, ["El archivo no tiene filas."]
    if len(tabla) > LIMITE_LOTE:
        return None, [f"El archivo tiene {len(tabla)} filas y el límite es {LIMITE_LOTE}."]
    d = tabla[COLUMNAS].copy()
    for c in NUMERICAS_VINO:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tipo_vino"] = d["tipo_vino"].astype(str).str.strip().str.lower().map(TIPOS)
    errores = []
    filas_malas = set()
    for c in NUMERICAS_VINO:
        e = ESQUEMA[c]
        malas = d.index[~d[c].between(e["minimo_valido"], e["maximo_valido"])]
        if len(malas):
            filas_malas.update(malas)
            errores.append(f"{e['etiqueta']}: {len(malas)} filas vacías o fuera del rango válido ({e['minimo_valido']:g} a {e['maximo_valido']:g}).")
    malas = d.index[d["tipo_vino"].isna()]
    if len(malas):
        filas_malas.update(malas)
        errores.append(f"Tipo de vino: {len(malas)} filas no dicen tinto ni blanco.")
    validas = d.drop(index=sorted(filas_malas))
    if validas.empty:
        return None, errores
    pipe, _ = modelo_final(CARPETAS[modelo])
    resultado = validas.copy()
    resultado.insert(0, "fila", validas.index.to_numpy() + 1)
    resultado.insert(1, "probabilidad_bueno", np.round(_probabilidad(pipe, validas), 4))
    return resultado.sort_values("probabilidad_bueno", ascending=False).reset_index(drop=True), errores
