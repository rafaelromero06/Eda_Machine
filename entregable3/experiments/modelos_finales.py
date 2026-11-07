import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.artefactos import buscar_final, guardar, reajustar
from src.benchmarks import MODELOS, NOMBRE_ORIGINAL, da_probabilidades
from src.data import cargar_conjunto
from src.original_model import auc_mann_whitney

RESULTADOS = RAIZ / "experiments" / "resultados"
TOLERANCIA_AUC = 0.001


def metricas_externas(modelo):
    cv = pd.read_csv(RESULTADOS / "cv_filas.csv")
    d = cv[(cv["conjunto"] == "vino") & (cv["modelo"] == modelo)]
    salida = {"n_pliegues": int(len(d)), "esquema": "validación anidada 5 x 3 repetida 5 veces (semillas 42-46)"}
    for m in ["auc", "pr_auc", "brier", "log_loss", "f1", "precision", "recall", "accuracy"]:
        if d[m].notna().any():
            salida[m] = {"media": float(d[m].mean()), "desviacion": float(d[m].std())}
    salida["tiempo_prediccion_ms_por_caso"] = float(d["tiempo_prediccion_ms_por_caso"].mean())
    return salida


def curva_oob_rf(pipe, X, y):
    rf = pipe.named_steps["modelo"]
    Xt = pipe.named_steps["prep"].transform(X)
    suma, cuenta, curva = np.zeros(len(y)), np.zeros(len(y)), []
    for t, (est, dentro) in enumerate(zip(rf.estimators_, rf.estimators_samples_), 1):
        fuera = np.ones(len(y), bool)
        fuera[dentro] = False
        suma[fuera] += est.predict_proba(Xt[fuera])[:, 1]
        cuenta[fuera] += 1
        if t % 25 == 0 or t == len(rf.estimators_):
            ok = cuenta > 0
            curva.append((t, auc_mann_whitney(y[ok], suma[ok] / cuenta[ok])))
    return curva


def reducir(modelo, pipe, hp):
    """Menor número de árboles con AUC fuera de bolsa a menos de TOLERANCIA_AUC del bosque completo."""
    datos = cargar_conjunto("vino")
    if modelo == "Random Forest":
        curva = curva_oob_rf(pipe, datos["X"], datos["y_evento"])
    else:
        curva = pipe.named_steps["modelo"].curva_oob_
    tope = curva[-1][1]
    T = next(t for t, a in curva if a >= tope - TOLERANCIA_AUC)
    return T, curva


def buscar_con_cache(modelo):
    """Búsqueda final; el resultado (hiperparámetros, puntaje y tiempo) se guarda para no repetirla."""
    from src.artefactos import slug
    cache = RESULTADOS / "busqueda_final" / f"{slug(modelo)}.json"
    if cache.exists():
        guardado = json.loads(cache.read_text(encoding="utf-8"))
        hp = {k: (float(v) if v in ("inf", "-inf") else v) for k, v in guardado["hiperparametros"].items()}
        pipe, tiempo = reajustar(modelo, hp)
        return hp, guardado["score_interno"], pipe, tiempo, guardado["tiempo_busqueda_s"]
    t0 = time.time()
    res = buscar_final(modelo)
    hp = dict(res.mejores_hiperparametros)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"modelo": modelo, "hiperparametros": {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                                                                        for k, v in hp.items()},
                                 "score_interno": res.mejor_score_interno, "tiempo_busqueda_s": round(time.time() - t0, 1)},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return hp, res.mejor_score_interno, res.mejor_estimador, res.tiempo_refit, round(time.time() - t0, 1)


def procesar(modelo):
    hp, score, pipe, tiempo, t_busqueda = buscar_con_cache(modelo)
    extra = {"tiempo_busqueda_final_s": t_busqueda}
    if modelo in ("Random Forest", NOMBRE_ORIGINAL):
        T, curva = reducir(modelo, pipe, hp)
        extra["reduccion_arboles"] = {"arboles_busqueda": int(hp["n_estimators"]), "arboles_desplegados": int(T),
                                      "criterio": f"menor número de árboles con AUC fuera de bolsa a menos de {TOLERANCIA_AUC} del bosque completo",
                                      "curva_auc_fuera_de_bolsa": [[int(t), float(a)] for t, a in curva]}
        if T < hp["n_estimators"]:
            pipe, tiempo = reajustar(modelo, hp, n_estimators=T)
    if modelo == NOMBRE_ORIGINAL:
        from src.artefactos import compactar_bof
        X_fijo = cargar_conjunto("vino")["X"]
        antes = pipe.predict_proba(X_fijo)[:, 1]
        compactar_bof(pipe.named_steps["modelo"])
        extra["compactacion"] = {"tipos": "hijos int32, valores y pesos float32, cortes float64",
                                 "diferencia_maxima_probabilidad": float(np.max(np.abs(antes - pipe.predict_proba(X_fijo)[:, 1])))}
    meta = guardar(modelo, pipe, hp, score, metricas_externas(modelo), tiempo, extra)
    return modelo, hp, score, meta["tamano_archivo_mb"], meta["prueba_ida_y_vuelta"]["identicas"]


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--solo-busqueda":
        buscar_con_cache(sys.argv[2])
        sys.exit(0)
    modelos = sys.argv[1:] or MODELOS
    filas = Parallel(n_jobs=min(len(modelos), 10), batch_size=1)(delayed(procesar)(m) for m in modelos)
    tabla = pd.DataFrame(filas, columns=["modelo", "hiperparametros", "score_interno", "tamano_mb", "ida_y_vuelta"])
    tabla["hiperparametros"] = tabla["hiperparametros"].map(lambda d: json.dumps({k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}, ensure_ascii=False))
    ruta = RESULTADOS / "hiperparametros_finales.csv"
    if ruta.exists():
        previa = pd.read_csv(ruta)
        tabla = pd.concat([previa[~previa["modelo"].isin(tabla["modelo"])], tabla], ignore_index=True)
    tabla.to_csv(ruta, index=False)
    print(tabla.to_string(index=False))
