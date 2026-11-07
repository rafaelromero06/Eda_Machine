import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.validacion import SEMILLAS_REPETICION

RESULTADOS = RAIZ / "experiments" / "resultados"
E2 = RAIZ.parent / "modelado_entregable2" / "modelado" / "experiments" / "tablas" / "tabla_maestra.csv"


def construir():
    e2 = pd.read_csv(E2)
    e2 = e2.assign(origen="Entregable 2", conjunto="vino", esquema="anidada 5 x 3", repeticiones=1, semillas="42",
                   pliegues=5, tiempo_busqueda_total_s=e2["tiempo_fold_s"] * 5)
    e2 = e2.rename(columns={"auc": "auc_media", "auc_std": "auc_sd", "f1": "f1_media", "brier": "brier_media",
                            "pr_auc": "pr_auc_media", "rmse": "rmse_media", "rmse_std": "rmse_sd", "mae": "mae_media"})
    cv = pd.read_csv(RESULTADOS / "cv_filas.csv")
    filas = []
    for (conjunto, modelo), d in cv.groupby(["conjunto", "modelo"]):
        reps = sorted(d["repeticion"].unique())
        filas.append({"origen": "Entregable 3", "conjunto": conjunto, "tarea": "clasificacion", "modelo": modelo,
                      "balanceo": d["balanceo"].iloc[0], "optimizador": d["optimizador"].iloc[0],
                      "esquema": "anidada 5 x 3" + (f" repetida {len(reps)} veces" if len(reps) > 1 else ""),
                      "repeticiones": len(reps), "semillas": ",".join(str(SEMILLAS_REPETICION[r]) for r in reps),
                      "pliegues": len(d), "auc_media": d["auc"].mean(), "auc_sd": d["auc"].std(),
                      "pr_auc_media": d["pr_auc"].mean(), "brier_media": d["brier"].mean(), "log_loss_media": d["log_loss"].mean(),
                      "f1_media": d["f1"].mean(), "precision": d["precision"].mean(), "recall": d["recall"].mean(),
                      "accuracy": d["accuracy"].mean(), "score_interno": d["score_interno"].mean(),
                      "n_evaluaciones": d["n_evaluaciones"].mean(), "tiempo_busqueda_total_s": d["tiempo_busqueda_s"].sum(),
                      "tiempo_ajuste_s": d["tiempo_ajuste_s"].mean(),
                      "tiempo_prediccion_ms_por_caso": d["tiempo_prediccion_ms_por_caso"].mean(),
                      "horas_cpu": d["tiempo_busqueda_s"].sum() / 3600})
    e3 = pd.DataFrame(filas)
    columnas = ["origen", "conjunto", "tarea", "modelo", "balanceo", "optimizador", "esquema", "repeticiones", "semillas", "pliegues",
                "auc_media", "auc_sd", "pr_auc_media", "brier_media", "log_loss_media", "f1_media", "rmse_media", "rmse_sd",
                "mae_media", "score_interno", "n_evaluaciones", "tiempo_busqueda_total_s", "tiempo_prediccion_ms_por_caso", "horas_cpu",
                "en_mcs", "dentro_1ee", "hiperparametros_tipicos"]
    tabla = pd.concat([e2, e3], ignore_index=True)
    tabla = tabla[[c for c in columnas if c in tabla.columns]]
    tabla.to_csv(RESULTADOS / "tabla_maestra_e3.csv", index=False)
    return tabla


if __name__ == "__main__":
    t = construir()
    print(t.groupby(["origen", "conjunto"]).size())
