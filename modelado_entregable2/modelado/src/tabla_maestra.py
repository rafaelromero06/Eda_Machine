from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import KFold

from . import config
from .estadistica import conjunto_confianza_modelos, delong, diebold_mariano, holm
from .evaluacion import ece
from .predicciones import cargar_oof, parsear_hiperparametros
from .seleccion import CLAVES, PRINCIPAL, candidatas_1ee, cargar_log, recomendada, resumir

RUTA_TABLA_MAESTRA = config.RUTA_EXPERIMENTOS / "tablas" / "tabla_maestra.csv"

DESCRIPCION = {
    "tarea": "clasificacion (quality ≥ 7) o regresion (quality)",
    "modelo / balanceo / optimizador": "la combinación evaluada",
    "rango": "puesto dentro de su tarea según la métrica principal (AUC o RMSE; 1 = mejor)",
    "auc, f1, recall, precision, accuracy": "media de los 5 folds externos (umbral 0.5); *_std = desviación entre folds",
    "rmse, mae, r2": "media de los 5 folds externos; *_std = desviación entre folds",
    "score_interno": "mejor score de la validación interna (AUC o RMSE) promediado entre folds",
    "brecha_interno_externo": "métrica externa − interna (AUC: negativo = optimista; RMSE: positivo = optimista)",
    "brier, ece, pr_auc": "calculadas con las 5,320 predicciones fuera de muestra (vacías para SVM, que no da probabilidades)",
    "dif_vs_mejor": "AUC (o error cuadrático medio) de la combinación menos el de la mejor de su tarea",
    "p_vs_mejor_holm": "p-valor de DeLong (clasificación) o Diebold-Mariano (regresión) contra la mejor, ajustado por Holm",
    "p_mcs, en_mcs": "p-valor del conjunto de confianza de modelos y si pertenece a él (α = 0.10; pérdida Brier o error cuadrático)",
    "dentro_1ee": "a menos de un error estándar de la mejor métrica principal",
    "recomendada": "configuración elegida con la regla de 1 error estándar",
    "n_evaluaciones": "configuraciones probadas por búsqueda (media de folds)",
    "tiempo_fold_s, horas_cpu": "segundos por fold externo y horas de CPU de las 5 búsquedas",
    "hiperparametros_tipicos": "mediana (numéricos) o moda (categóricos) de lo elegido en los 5 folds",
}


def hiperparametros_tipicos(textos) -> str:
    """Resumen de los hiperparámetros elegidos en los folds: mediana o valor más frecuente."""
    dicts = [parsear_hiperparametros(t) for t in textos]
    partes = []
    for clave in dicts[0]:
        valores = [d[clave] for d in dicts if clave in d]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in valores):
            partes.append(f"{clave}={int(round(np.median(valores)))}")
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valores):
            partes.append(f"{clave}={float(np.median(valores)):.3g}")
        else:
            valor, veces = Counter(map(str, valores)).most_common(1)[0]
            partes.append(f"{clave}={valor}" + (f" ({veces}/5)" if veces < len(valores) else ""))
    return ", ".join(partes)


def _clave(df: pd.DataFrame) -> pd.Series:
    return df["modelo"] + "|" + df["balanceo"] + "|" + df["optimizador"]


def _r2_por_fold(log: pd.DataFrame, y_reg: pd.Series) -> pd.Series:
    """R² exacto de cada fold: 1 − RMSE² / Var(y de prueba)."""
    folds = KFold(n_splits=config.K_OUTER, shuffle=True, random_state=config.RANDOM_STATE).split(y_reg)
    var_test = {i: y_reg.iloc[te].var(ddof=0) for i, (_, te) in enumerate(folds)}
    return 1 - log["rmse"] ** 2 / log["fold_externo"].map(var_test)


def _ancha(oof: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Predicciones OOF en formato ancho: una columna por combinación, filas ordenadas por idx."""
    oof = oof.assign(clave=_clave(oof))
    return oof.pivot(index="idx", columns="clave", values=columna)


def _indicadores_clasificacion(oof: pd.DataFrame, mejor: str, alfa: float, B: int) -> pd.DataFrame:
    y = oof.drop_duplicates("idx").set_index("idx")["y_true"].sort_index().to_numpy()
    puntajes = _ancha(oof, "score")
    tipos = oof.assign(clave=_clave(oof)).groupby("clave")["tipo_score"].first()
    filas = []
    for clave in puntajes.columns:
        s = puntajes[clave].to_numpy()
        fila = {"clave": clave}
        if tipos[clave] == "probabilidad":
            fila.update(brier=brier_score_loss(y, s), ece=ece(y, s), pr_auc=average_precision_score(y, s))
        if clave != mejor:
            d = delong(y, s, puntajes[mejor].to_numpy())
            fila.update(dif_vs_mejor=d["diferencia"], p_vs_mejor=d["p"])
        else:
            fila.update(dif_vs_mejor=0.0, p_vs_mejor=np.nan)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    probabilisticas = [c for c in puntajes.columns if tipos[c] == "probabilidad"]
    perdidas = pd.DataFrame({c: (y - puntajes[c].to_numpy()) ** 2 for c in probabilisticas})
    mcs = conjunto_confianza_modelos(perdidas, alfa=alfa, B=B).rename(columns={"modelo": "clave"})
    return tabla.merge(mcs[["clave", "p_mcs", "en_mcs"]], on="clave", how="left")


def _indicadores_regresion(oof: pd.DataFrame, mejor: str, alfa: float, B: int) -> pd.DataFrame:
    y = oof.drop_duplicates("idx").set_index("idx")["y_true"].sort_index().to_numpy()
    pred = _ancha(oof, "pred")
    errores = {c: y - pred[c].to_numpy() for c in pred.columns}
    filas = []
    for clave, e in errores.items():
        fila = {"clave": clave}
        if clave != mejor:
            d = diebold_mariano(e, errores[mejor])
            fila.update(dif_vs_mejor=d["diferencia"], p_vs_mejor=d["p"])
        else:
            fila.update(dif_vs_mejor=0.0, p_vs_mejor=np.nan)
        filas.append(fila)
    mcs = conjunto_confianza_modelos(pd.DataFrame({c: e ** 2 for c, e in errores.items()}), alfa=alfa, B=B)
    mcs = mcs.rename(columns={"modelo": "clave"})
    return pd.DataFrame(filas).merge(mcs[["clave", "p_mcs", "en_mcs"]], on="clave", how="left")


def construir(alfa_mcs: float = 0.10, B: int = 2000) -> pd.DataFrame:
    """Arma la tabla maestra a partir de los logs y de las predicciones OOF."""
    from .data_prep import preparar_dataset, separar_X_y

    _, y_reg, _ = separar_X_y(preparar_dataset(verbose=False))
    partes = []
    for tarea in ["clasificacion", "regresion"]:
        log = cargar_log(tarea)
        if tarea == "regresion":
            log = log.assign(r2=_r2_por_fold(log, y_reg))
        resumen = resumir(log, tarea)
        if tarea == "regresion":
            g = log.groupby(CLAVES)["r2"]
            resumen = resumen.merge(g.mean().rename("r2").reset_index(), on=CLAVES)
        metrica, mayor = PRINCIPAL[tarea]
        resumen["rango"] = resumen[metrica].rank(ascending=not mayor, method="min").astype(int)
        interno = resumen["score_interno"] if mayor else -resumen["score_interno"]
        resumen["score_interno"] = interno
        resumen["brecha_interno_externo"] = resumen[metrica] - interno
        horas = log.groupby(CLAVES)["tiempo_seg"].sum().div(3600).rename("horas_cpu").reset_index()
        resumen = resumen.merge(horas, on=CLAVES)
        resumen["hiperparametros_tipicos"] = (log.sort_values("fold_externo").groupby(CLAVES)["mejores_hiperparametros"]
                                              .apply(hiperparametros_tipicos).reindex(pd.MultiIndex.from_frame(resumen[CLAVES]))
                                              .to_numpy())
        resumen["dentro_1ee"] = _clave(resumen).isin(_clave(candidatas_1ee(resumen, tarea)))
        elegida = recomendada(resumen, tarea)
        resumen["recomendada"] = _clave(resumen) == "|".join(elegida[CLAVES])
        mejor = _clave(resumen.loc[[resumen[metrica].idxmax() if mayor else resumen[metrica].idxmin()]]).iloc[0]
        oof = cargar_oof(tarea)
        indicadores = (_indicadores_clasificacion if tarea == "clasificacion" else _indicadores_regresion)(
            oof, mejor, alfa_mcs, B)
        indicadores["p_vs_mejor_holm"] = np.nan
        validos = indicadores["p_vs_mejor"].notna()
        indicadores.loc[validos, "p_vs_mejor_holm"] = holm(indicadores.loc[validos, "p_vs_mejor"])
        resumen = resumen.assign(clave=_clave(resumen)).merge(indicadores, on="clave", how="left")
        resumen.insert(0, "tarea", tarea)
        partes.append(resumen.drop(columns="clave"))

    tabla = pd.concat(partes, ignore_index=True)
    orden = ["tarea", "modelo", "balanceo", "optimizador", "rango",
             "auc", "auc_std", "f1", "f1_std", "recall", "recall_std", "precision", "precision_std",
             "accuracy", "accuracy_std", "rmse", "rmse_std", "mae", "mae_std", "r2",
             "score_interno", "brecha_interno_externo", "brier", "ece", "pr_auc",
             "dif_vs_mejor", "p_vs_mejor", "p_vs_mejor_holm", "p_mcs", "en_mcs", "dentro_1ee", "recomendada",
             "n_evaluaciones", "tiempo_fold_s", "horas_cpu", "hiperparametros_tipicos"]
    tabla = tabla[[c for c in orden if c in tabla.columns]]
    return tabla.sort_values(["tarea", "rango"]).reset_index(drop=True)
