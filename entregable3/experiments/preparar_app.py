import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.artefactos import cargar_artefacto, slug
from src.benchmarks import BENCHMARKS_CLASE, MODELOS, NOMBRE_ORIGINAL
from src.data import CATEGORICAS_VINO, DICCIONARIO_VINO, NUMERICAS_VINO, RUTA_APP, cargar_vinos
from src.explicaciones import Explicador
from src.original_model import auc_mann_whitney

RESULTADOS = RAIZ / "experiments" / "resultados"
TABLAS = RESULTADOS / "tablas"
LITERATURA = RAIZ / "experiments" / "revision_literatura"
E2 = RAIZ.parent / "modelado_entregable2" / "modelado" / "experiments" / "tablas"
DESPLEGABLES = [NOMBRE_ORIGINAL] + BENCHMARKS_CLASE + ["LightGBM", "CatBoost"]
COPIAR = ["calibracion.csv", "calibracion_curvas.csv", "rangos_friedman.csv", "friedman.json", "tabla_pruebas.csv",
          "pruebas_bof.csv", "ablacion.csv", "adicionales_pruebas.csv", "adicionales_auc.csv", "veredicto.json",
          "robustez_sensibilidad.csv", "robustez_referencia.csv", "robustez_resumen.csv", "errores_subgrupos.csv",
          "bootstrap_bca_mejor.json"]
PAREJAS_POR_CLASE = 200


def umbral_conforme(no_conformidad, alfa):
    n = len(no_conformidad)
    return float(np.quantile(no_conformidad, min(1.0, np.ceil((n + 1) * (1 - alfa)) / n), method="higher"))


def es_final(modelo):
    """True si el artefacto del modelo existe y no es uno de los provisionales de prueba."""
    ruta = RAIZ / "artifacts" / slug(modelo) / "metadata.json"
    if not ruta.exists():
        return False
    return not json.loads(ruta.read_text(encoding="utf-8"))["metricas_bucle_externo"].get("simulado", False)


def probabilidad(modelo, puntaje):
    return puntaje if modelo != "SVM" else 1 / (1 + np.exp(-puntaje))


def copiar_tablas():
    for nombre in COPIAR:
        if (TABLAS / nombre).exists():
            shutil.copy(TABLAS / nombre, RUTA_APP / nombre)
        else:
            (RUTA_APP / nombre).unlink(missing_ok=True)


def lideres():
    resumen = pd.read_csv(TABLAS / "resumen_modelos.csv")
    tam = {m: json.loads((RAIZ / "artifacts" / slug(m) / "metadata.json").read_text(encoding="utf-8"))["tamano_archivo_mb"]
           for m in DESPLEGABLES if es_final(m)}
    resumen["tamano_mb"] = resumen["modelo"].map(tam)
    resumen.to_csv(RUTA_APP / "lideres.csv", index=False)


def umbrales_conformes(ancho, y):
    conforme = pd.read_csv(TABLAS / "robustez_conforme.csv") if (TABLAS / "robustez_conforme.csv").exists() else None
    salida = {}
    for m in MODELOS:
        p = probabilidad(m, ancho[m].to_numpy())
        nc = np.where(y == 1, 1 - p, p)
        salida[m] = {"0.1": umbral_conforme(nc, 0.10), "0.05": umbral_conforme(nc, 0.05)}
        if conforme is not None:
            c = conforme[(conforme["modelo"] == m) & (conforme["tipo"] == "marginal")].set_index("alfa")["cobertura"]
            salida[m].update({"cobertura_0.1": float(c.get(0.10, np.nan)), "cobertura_0.05": float(c.get(0.05, np.nan))})
    (RUTA_APP / "umbrales_conformes.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")


def decision(ancho, y):
    """Vinos buenos y no buenos que superan cada umbral, para explorar la decisión en el navegador."""
    rejilla = np.round(np.linspace(0, 1, 101), 2)
    modelos = {}
    for m in MODELOS:
        p = probabilidad(m, ancho[m].to_numpy())
        modelos[m] = {"vp": [int(((p >= t) & (y == 1)).sum()) for t in rejilla],
                      "fp": [int(((p >= t) & (y == 0)).sum()) for t in rejilla]}
    salida = {"umbrales": rejilla.tolist(), "positivos": int(y.sum()), "negativos": int((1 - y).sum()), "modelos": modelos}
    (RUTA_APP / "decision.json").write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")


def cata(ancho, y, notas):
    """Vinos reales con la probabilidad fuera de muestra del BOF para el juego de la cata a ciegas."""
    vinos = cargar_vinos().iloc[ancho.index.to_numpy()]
    rng = np.random.default_rng(2026)
    buenos = rng.choice(np.flatnonzero(y == 1), PAREJAS_POR_CLASE, replace=False)
    malos = rng.choice(np.flatnonzero(y == 0), PAREJAS_POR_CLASE, replace=False)
    fila = lambda i: {"v": [round(float(vinos[c].iloc[i]), 4) for c in NUMERICAS_VINO], "t": vinos["tipo_vino"].iloc[i],
                      "n": int(notas[i]), "p": round(float(ancho[NOMBRE_ORIGINAL].iloc[i]), 4)}
    escala = [{"etiqueta": DICCIONARIO_VINO[c][0], "unidad": DICCIONARIO_VINO[c][1],
               "bajo": float(cargar_vinos()[c].quantile(0.02)), "alto": float(cargar_vinos()[c].quantile(0.98))}
              for c in NUMERICAS_VINO]
    salida = {"buenos": [fila(i) for i in buenos], "malos": [fila(i) for i in malos], "escala": escala,
              "auc": float(auc_mann_whitney(y, ancho[NOMBRE_ORIGINAL].to_numpy()))}
    (RUTA_APP / "cata.json").write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")


def entregable2():
    """Resultados del Entregable 2 que el tablero muestra como antecedente."""
    if not E2.exists():
        return
    m = pd.read_csv(E2 / "tabla_maestra.csv")
    clas = m[m["tarea"] == "clasificacion"][["modelo", "balanceo", "optimizador", "auc", "auc_std", "f1", "pr_auc", "brier", "en_mcs",
                                             "tiempo_fold_s"]]
    clas.to_csv(RUTA_APP / "e2_clasificacion.csv", index=False)
    reg = m[m["tarea"] == "regresion"][["modelo", "optimizador", "rmse", "rmse_std", "mae", "r2", "tiempo_fold_s"]]
    reg.to_csv(RUTA_APP / "e2_regresion.csv", index=False)
    shutil.copy(E2 / "tabla21_importancia_shap.csv", RUTA_APP / "e2_shap.csv")
    shutil.copy(E2 / "tabla05_optimizadores.csv", RUTA_APP / "e2_optimizadores.csv")
    shutil.copy(E2 / "tabla03_clf_efecto_balanceo.csv", RUTA_APP / "e2_balanceo.csv")


def revision():
    """Cifras de la revisión de literatura que respaldan la declaración de novedad."""
    matriz = pd.read_csv(LITERATURA / "matriz_novedad.csv")
    fecha = max(LITERATURA.glob("busqueda_*.json")).stem.split("_")[-1]
    citas = json.loads(max(LITERATURA.glob("citas_*.json")).read_text(encoding="utf-8"))
    repos = json.loads(max(LITERATURA.glob("repositorios_*.json")).read_text(encoding="utf-8"))
    titulos = {p.get("title") for s in citas["semillas"].values() for t in ("citas", "referencias") for p in s[t] if p.get("title")}
    salida = {"fecha": fecha, "bases": int(matriz["base"].nunique()), "cadenas": int(matriz["cadena"].nunique()),
              "revisados": int(matriz["revisados"].sum()), "relevantes": int(matriz["relevantes"].sum()),
              "semillas": len(citas["semillas"]), "citacion": len(titulos), "consultas_github": len(repos["consultas"])}
    (RUTA_APP / "revision.json").write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")


def shap_global():
    vinos = cargar_vinos()
    fondo = vinos.sample(200, random_state=0)[NUMERICAS_VINO + CATEGORICAS_VINO]
    muestra = vinos.sample(300, random_state=1)[NUMERICAS_VINO + CATEGORICAS_VINO]
    filas = []
    for m in DESPLEGABLES:
        if not es_final(m):
            continue
        pipe, _ = cargar_artefacto(m)
        exp = Explicador(m, pipe, fondo)
        n = 300 if exp.modo != "permutacion" else 120
        imp = exp.explicar_varios(muestra.iloc[:n])[0].abs().mean()
        for v, i in imp.items():
            filas.append({"modelo": m, "variable": v, "importancia": float(i), "escala": exp.escala, "n_vinos": n})
        print("SHAP global:", m, flush=True)
    ruta = RUTA_APP / "shap_global.csv"
    if filas:
        pd.DataFrame(filas).to_csv(ruta, index=False)
    else:
        ruta.unlink(missing_ok=True)


def main():
    RUTA_APP.mkdir(parents=True, exist_ok=True)
    copiar_tablas()
    lideres()
    filas = pd.read_csv(RESULTADOS / "cv_filas.csv")
    vino = filas[filas["conjunto"] == "vino"]
    vino.pivot_table(index=["repeticion", "fold"], columns="modelo", values="auc")[MODELOS].reset_index().to_csv(
        RUTA_APP / "bloques_auc.csv", index=False)
    pred = pd.read_csv(RESULTADOS / "cv_predicciones.csv")
    p0 = pred[(pred["conjunto"] == "vino") & (pred["repeticion"] == 0)]
    ancho = p0.pivot_table(index="idx", columns="modelo", values="puntaje")[MODELOS]
    meta = p0.drop_duplicates("idx").set_index("idx").loc[ancho.index, ["y_evento", "y_ord"]]
    pd.concat([meta, ancho], axis=1).reset_index().to_csv(RUTA_APP / "oof_rep1.csv.gz", index=False, compression="gzip")
    y = meta["y_evento"].to_numpy()
    umbrales_conformes(ancho, y)
    decision(ancho, y)
    cata(ancho, y, meta["y_ord"].to_numpy())
    entregable2()
    revision()
    shap_global()
    print("datos del dashboard listos en", RUTA_APP)


if __name__ == "__main__":
    main()
