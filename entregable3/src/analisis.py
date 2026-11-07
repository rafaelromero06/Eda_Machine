from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .benchmarks import BENCHMARKS_CLASE, BENCHMARKS_SOTA, CORTOS, MEJORES, MODELOS, NOMBRE_ORIGINAL
from .data import RUTA_BASE, cargar_conjunto
from .original_model import auc_mann_whitney
from .pruebas import (bayes_rangos_signo, bayes_t_correlada, bootstrap_bca_dif_auc, d_cohen_pareada, delong,
                      delta_cliff, friedman, holm, nemenyi_p, t_corregida, wilcoxon_pareado)

RESULTADOS = RUTA_BASE / "experiments" / "resultados"
TABLAS = RESULTADOS / "tablas"
BENCHMARKS = BENCHMARKS_CLASE + BENCHMARKS_SOTA
ALFA, ROPE_VINO, ROPE_CONJUNTOS = 0.05, 0.005, 0.01
CONJUNTOS_ADICIONALES = ["abalone", "cardiotocografia", "estudiantes", "corazon"]


def cargar():
    filas = pd.read_csv(RESULTADOS / "cv_filas.csv")
    pred = pd.read_csv(RESULTADOS / "cv_predicciones.csv")
    return filas, pred


def tabla_bloques(filas, conjunto="vino", metrica="auc"):
    d = filas[filas["conjunto"] == conjunto]
    return d.pivot_table(index=["repeticion", "fold"], columns="modelo", values=metrica)[MODELOS]


def oof(pred, conjunto="vino", repeticion=0):
    d = pred[(pred["conjunto"] == conjunto) & (pred["repeticion"] == repeticion)]
    ancho = d.pivot_table(index="idx", columns="modelo", values="puntaje")
    y = d.drop_duplicates("idx").set_index("idx")["y_evento"].loc[ancho.index]
    return y.to_numpy(), ancho


def ic_bootstrap_auc(y, s, B=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        vals.append(auc_mann_whitney(y[idx], s[idx]))
    return np.quantile(vals, [0.025, 0.975])


def ece(y, p, grupos=10):
    """Error de calibración esperado con grupos por cuantiles de la probabilidad predicha."""
    cortes = np.unique(np.quantile(p, np.linspace(0, 1, grupos + 1)))
    g = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, len(cortes) - 2)
    total = 0.0
    for k in np.unique(g):
        m = g == k
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def resumen_modelos(filas, pred):
    y, ancho = oof(pred)
    salida = []
    for m in MODELOS:
        d = filas[(filas["conjunto"] == "vino") & (filas["modelo"] == m)]
        lo, hi = ic_bootstrap_auc(y, ancho[m].to_numpy())
        salida.append({
            "modelo": m, "corto": CORTOS[m], "balanceo": MEJORES[m][0], "optimizador": MEJORES[m][1],
            "auc_media": d["auc"].mean(), "auc_sd": d["auc"].std(), "auc_oof_rep1": auc_mann_whitney(y, ancho[m].to_numpy()),
            "auc_ic_inf": lo, "auc_ic_sup": hi, "pr_auc_media": d["pr_auc"].mean(), "brier_media": d["brier"].mean(),
            "log_loss_media": d["log_loss"].mean(), "f1_media": d["f1"].mean(), "precision_media": d["precision"].mean(),
            "recall_media": d["recall"].mean(), "evaluaciones_media": d["n_evaluaciones"].mean(),
            "busqueda_s_media": d["tiempo_busqueda_s"].mean(), "ajuste_s_media": d["tiempo_ajuste_s"].mean(),
            "prediccion_ms_por_caso": d["tiempo_prediccion_ms_por_caso"].mean(), "pliegues": len(d)})
    return pd.DataFrame(salida).sort_values("auc_media", ascending=False).reset_index(drop=True)


def pruebas_bof(filas, pred):
    """Pruebas P1-P4 y tamaños de efecto del BOF frente a cada benchmark (vino)."""
    tab = tabla_bloques(filas)
    N = len(tab)
    n_test = filas.loc[filas["conjunto"] == "vino", "n_test"].mean()
    n_train = filas.loc[filas["conjunto"] == "vino", "n_train"].mean()
    fr = friedman(tab)
    nem = nemenyi_p(fr["rangos"], N)
    y, ancho = oof(pred)
    bof = tab[NOMBRE_ORIGINAL].to_numpy()
    filas_p = []
    for b in BENCHMARKS:
        otro = tab[b].to_numpy()
        dif = bof - otro
        w = wilcoxon_pareado(bof, otro)
        t = t_corregida(dif, n_train, n_test)
        dl = delong(y, ancho[NOMBRE_ORIGINAL].to_numpy(), ancho[b].to_numpy())
        cliff, mag = delta_cliff(bof, otro)
        bt = bayes_t_correlada(dif, n_test / (n_test + n_train), ROPE_VINO)
        filas_p.append({"benchmark": b, "auc_bof": bof.mean(), "auc_benchmark": otro.mean(), "diferencia_media": dif.mean(),
                        "bloques_a_favor": int((dif > 0).sum()), "nemenyi_p": float(nem.loc[NOMBRE_ORIGINAL, b]),
                        "wilcoxon_W": w["estadistico"], "wilcoxon_p": w["p"], "t_corregida": t["estadistico"],
                        "t_p": t["p"], "t_ic_inf": t["ic_inf"], "t_ic_sup": t["ic_sup"], "delong_z": dl["estadistico"],
                        "delong_p": dl["p"], "delong_dif": dl["diferencia"], "delong_ic_inf": dl["ic_inf"],
                        "delong_ic_sup": dl["ic_sup"], "delta_cliff": cliff, "magnitud_cliff": mag,
                        "d_cohen_pareada": d_cohen_pareada(dif), "rank_biserial": w["rank_biserial"],
                        "bayes_p_peor": bt["p_peor"], "bayes_p_equivalente": bt["p_equivalente"], "bayes_p_mejor": bt["p_mejor"]})
    p = pd.DataFrame(filas_p)
    for col in ("wilcoxon_p", "t_p", "delong_p"):
        p[col.replace("_p", "_p_holm")] = holm(p[col])
    todos = holm(np.concatenate([p["wilcoxon_p"], p["t_p"], p["delong_p"]]))
    p["wilcoxon_p_holm27"], p["t_p_holm27"], p["delong_p_holm27"] = np.split(todos, 3)
    return p.sort_values("auc_benchmark", ascending=False).reset_index(drop=True), fr, nem


def delong_por_repeticion(pred):
    salida = []
    for r in sorted(pred["repeticion"].unique()):
        y, ancho = oof(pred, "vino", r)
        if NOMBRE_ORIGINAL not in ancho:
            continue
        ps = [delong(y, ancho[NOMBRE_ORIGINAL].to_numpy(), ancho[b].to_numpy()) for b in BENCHMARKS]
        aj = holm([d["p"] for d in ps])
        for b, d, a in zip(BENCHMARKS, ps, aj):
            salida.append({"repeticion": r + 1, "benchmark": b, "diferencia": d["diferencia"], "z": d["estadistico"],
                           "p": d["p"], "p_holm": a})
    return pd.DataFrame(salida)


def bootstrap_frente_al_mejor(pred, mejor):
    y, ancho = oof(pred)
    return bootstrap_bca_dif_auc(y, ancho[NOMBRE_ORIGINAL].to_numpy(), ancho[mejor].to_numpy())


def tabla_pruebas_formal(p, fr, mejor, boot):
    """Tabla de pruebas con el formato pedido: prueba, H0, supuestos, estadístico, p crudo, p ajustado, efecto, conclusión."""
    f = p.set_index("benchmark").loc[mejor]
    concl = lambda pv, dif: ("se rechaza H0 a favor del BOF" if pv < ALFA and dif > 0 else
                             "se rechaza H0 en contra del BOF" if pv < ALFA else "no se rechaza H0")
    return pd.DataFrame([
        {"prueba": "P1. Friedman (Iman–Davenport)", "hipotesis_nula": "Los 10 modelos tienen el mismo rango medio",
         "supuestos": "Bloques independientes (aproximado: pliegues repetidos comparten casos)",
         "estadistico": f"F = {fr['F_ID']:.2f} ({fr['k'] - 1}, {(fr['k'] - 1) * (fr['N'] - 1)} gl)", "p_crudo": fr["p_ID"],
         "p_ajustado": fr["p_ID"], "efecto": f"W de Kendall = {fr['W']:.2f}",
         "conclusion": "hay diferencias entre modelos" if fr["p_ID"] < ALFA else "sin diferencias"},
        {"prueba": f"P1. Nemenyi (BOF vs {CORTOS[mejor]})", "hipotesis_nula": "Mismo rango medio",
         "supuestos": "Post-hoc de Friedman, todos los pares", "estadistico": f"CD = {fr['cd']:.2f}",
         "p_crudo": f["nemenyi_p"], "p_ajustado": f["nemenyi_p"], "efecto": f"δ de Cliff = {f['delta_cliff']:.2f}",
         "conclusion": concl(f["nemenyi_p"], f["diferencia_media"])},
        {"prueba": f"P2. Wilcoxon (BOF vs {CORTOS[mejor]})", "hipotesis_nula": "Mediana de la diferencia de AUC = 0",
         "supuestos": "Diferencias pareadas simétricas; 25 bloques", "estadistico": f"W = {f['wilcoxon_W']:.0f}",
         "p_crudo": f["wilcoxon_p"], "p_ajustado": f["wilcoxon_p_holm"], "efecto": f"rango-biserial = {f['rank_biserial']:.2f}",
         "conclusion": concl(f["wilcoxon_p_holm"], f["diferencia_media"])},
        {"prueba": f"P3. t corregida de Nadeau–Bengio (BOF vs {CORTOS[mejor]})", "hipotesis_nula": "Media de la diferencia de AUC = 0",
         "supuestos": "Normalidad aproximada de las diferencias; varianza corregida por solapamiento",
         "estadistico": f"t = {f['t_corregida']:.2f} (24 gl)", "p_crudo": f["t_p"], "p_ajustado": f["t_p_holm"],
         "efecto": f"d pareada = {f['d_cohen_pareada']:.2f}", "conclusion": concl(f["t_p_holm"], f["diferencia_media"])},
        {"prueba": f"P4. DeLong (BOF vs {CORTOS[mejor]})", "hipotesis_nula": "AUC iguales sobre los mismos 5320 vinos",
         "supuestos": "Casos independientes; predicciones fuera de muestra de la repetición 1",
         "estadistico": f"z = {f['delong_z']:.2f}", "p_crudo": f["delong_p"], "p_ajustado": f["delong_p_holm"],
         "efecto": f"ΔAUC = {f['delong_dif']:.4f} [{boot['ic_inf']:.4f}, {boot['ic_sup']:.4f}] (BCa)",
         "conclusion": concl(f["delong_p_holm"], f["delong_dif"])},
    ])


def ablacion(filas_cv):
    ab = pd.read_csv(RESULTADOS / "ablacion_filas.csv")
    pr = pd.read_csv(RESULTADOS / "ablacion_predicciones.csv.gz")
    salida = []
    for conjunto, d in ab.groupby("conjunto"):
        ancho = d.pivot_table(index=["repeticion", "fold"], columns="variante", values="auc")
        completo = ancho["completo"].to_numpy()
        n_test = filas_cv.loc[filas_cv["conjunto"] == conjunto, "n_test"].mean()
        n_train = filas_cv.loc[filas_cv["conjunto"] == conjunto, "n_train"].mean()
        p0 = pr[(pr["conjunto"] == conjunto) & (pr["repeticion"] == 0)].pivot_table(index="idx", columns="variante", values="puntaje")
        y0 = pr[(pr["conjunto"] == conjunto) & (pr["repeticion"] == 0)].drop_duplicates("idx").set_index("idx")["y_evento"].loc[p0.index].to_numpy()
        for v in ancho.columns:
            dv = d[d["variante"] == v]
            fila = {"conjunto": conjunto, "variante": v, "auc_media": dv["auc"].mean(), "auc_sd": dv["auc"].std(),
                    "brier_media": dv["brier"].mean(), "log_loss_media": dv["log_loss"].mean(),
                    "diferencia_vs_completo": (completo - ancho[v].to_numpy()).mean()}
            if v != "completo" and len(completo) >= 5:
                dif = completo - ancho[v].to_numpy()
                fila["bloques_a_favor_completo"] = int((dif > 0).sum())
                if np.any(dif != 0):
                    fila["wilcoxon_p"] = wilcoxon_pareado(completo, ancho[v].to_numpy())["p"]
                    fila["t_p"] = t_corregida(dif, n_train, n_test)["p"]
                    fila["delong_p"] = delong(y0, p0["completo"].to_numpy(), p0[v].to_numpy())["p"]
            salida.append(fila)
    t = pd.DataFrame(salida)
    for conjunto, idx in t.groupby("conjunto").groups.items():
        sub = t.loc[idx]
        m = sub["variante"].ne("completo") & sub["wilcoxon_p"].notna()
        for col in ("wilcoxon_p", "t_p", "delong_p"):
            t.loc[sub.index[m], col + "_holm"] = holm(sub.loc[m, col])
    return t


def conjuntos_adicionales(filas, pred):
    salida, pruebas = [], []
    for c in CONJUNTOS_ADICIONALES:
        d = filas[filas["conjunto"] == c]
        if d.empty:
            continue
        medias = d.groupby("modelo")["auc"].agg(["mean", "std"])
        for m, r in medias.iterrows():
            salida.append({"conjunto": c, "modelo": m, "auc_media": r["mean"], "auc_sd": r["std"],
                           "brier_media": d.loc[d["modelo"] == m, "brier"].mean()})
        mejor = medias.drop(index=NOMBRE_ORIGINAL).sort_values("mean").index[-1]
        y, ancho = oof(pred, c, 0)
        dl = delong(y, ancho[NOMBRE_ORIGINAL].to_numpy(), ancho[mejor].to_numpy())
        pruebas.append({"conjunto": c, "mejor_benchmark": mejor, "auc_bof": medias.loc[NOMBRE_ORIGINAL, "mean"],
                        "auc_mejor": medias.loc[mejor, "mean"], "bof_mejor_en_media": bool(medias.loc[NOMBRE_ORIGINAL, "mean"] > medias.loc[mejor, "mean"]),
                        "puesto_bof": int(medias["mean"].rank(ascending=False)[NOMBRE_ORIGINAL]),
                        "delong_dif": dl["diferencia"], "delong_p": dl["p"]})
    pr = pd.DataFrame(pruebas)
    if not pr.empty:
        pr["delong_p_holm"] = holm(pr["delong_p"])
        pr["significativamente_peor"] = (pr["delong_p_holm"] < ALFA) & (pr["delong_dif"] < 0)
    return pd.DataFrame(salida), pr


def entre_conjuntos(filas, adic_tabla):
    vino = filas[filas["conjunto"] == "vino"].groupby("modelo")["auc"].mean().rename("vino")
    otros = adic_tabla.pivot_table(index="modelo", columns="conjunto", values="auc_media")
    tab = pd.concat([vino, otros], axis=1).T[MODELOS].dropna()
    fr = friedman(tab)
    pares = []
    for b in BENCHMARKS:
        dif = (tab[NOMBRE_ORIGINAL] - tab[b]).to_numpy()
        w = stats.wilcoxon(dif) if np.any(dif != 0) else None
        bs = bayes_rangos_signo(dif, ROPE_CONJUNTOS)
        pares.append({"benchmark": b, "conjuntos_a_favor": int((dif > 0).sum()), "diferencia_media": dif.mean(),
                      "wilcoxon_p": float(w.pvalue) if w is not None else 1.0, **{f"bayes_{k}": v for k, v in bs.items()}})
    pares = pd.DataFrame(pares)
    pares["wilcoxon_p_holm"] = holm(pares["wilcoxon_p"])
    mejor_por_conjunto = tab.drop(columns=NOMBRE_ORIGINAL).max(axis=1)
    bs_mejor = bayes_rangos_signo((tab[NOMBRE_ORIGINAL] - mejor_por_conjunto).to_numpy(), ROPE_CONJUNTOS)
    return tab, fr, pares, bs_mejor


def calibracion(pred):
    y, ancho = oof(pred)
    filas, curvas = [], []
    for m in MODELOS:
        if m == "SVM":
            continue
        p = ancho[m].to_numpy()
        filas.append({"modelo": m, "ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
                      "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))),
                      "media_predicha": float(p.mean()), "prevalencia": float(y.mean())})
        cortes = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        g = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, len(cortes) - 2)
        for k in np.unique(g):
            sel = g == k
            curvas.append({"modelo": m, "grupo": int(k), "p_media": float(p[sel].mean()), "frecuencia": float(y[sel].mean()),
                           "n": int(sel.sum())})
    return pd.DataFrame(filas).sort_values("ece"), pd.DataFrame(curvas)


def errores(pred):
    """AUC por subgrupo y probabilidad media por nota real (repetición 1)."""
    datos = cargar_conjunto("vino")
    X = datos["X"]
    y, ancho = oof(pred)
    idx = ancho.index.to_numpy()
    tipo = X["tipo_vino"].to_numpy()[idx]
    nota = datos["y_ord"][idx]
    alcohol = X["alcohol"].to_numpy()[idx]
    tramos = pd.qcut(alcohol, 3, labels=["alcohol bajo", "alcohol medio", "alcohol alto"]).astype(str)
    grupos = {"todos": np.ones(len(y), bool), "tinto": tipo == "red", "blanco": tipo == "white",
              **{t: tramos == t for t in ["alcohol bajo", "alcohol medio", "alcohol alto"]},
              "frontera (nota 6 o 7)": np.isin(nota, [6, 7])}
    salida = []
    for g, m in grupos.items():
        for mod in [NOMBRE_ORIGINAL, "Random Forest", "CatBoost", "XGBoost", "LightGBM"]:
            s = ancho[mod].to_numpy()
            salida.append({"grupo": g, "modelo": mod, "n": int(m.sum()), "prevalencia": float(y[m].mean()),
                           "auc": auc_mann_whitney(y[m], s[m]) if 0 < y[m].sum() < m.sum() else np.nan})
    por_nota = pd.DataFrame({"nota": nota, **{CORTOS[m]: ancho[m].to_numpy() for m in [NOMBRE_ORIGINAL, "Random Forest"]}})
    por_nota = por_nota.groupby("nota").agg(["mean", "count"])
    rango_bof = ancho[NOMBRE_ORIGINAL].rank(pct=True).to_numpy()
    rango_rf = ancho["Random Forest"].rank(pct=True).to_numpy()
    casos = pd.DataFrame({"idx": idx, "nota": nota, "tipo": tipo, "y": y, "rango_bof": rango_bof, "rango_rf": rango_rf})
    casos["mejora_bof"] = np.where(casos["y"] == 1, casos["rango_bof"] - casos["rango_rf"], casos["rango_rf"] - casos["rango_bof"])
    return pd.DataFrame(salida), por_nota, casos


def veredicto(resumen, p, fr, adic_pr):
    mejor = resumen[resumen["modelo"] != NOMBRE_ORIGINAL].iloc[0]["modelo"]
    f = p.set_index("benchmark").loc[mejor]
    auc_bof = resumen.set_index("modelo").loc[NOMBRE_ORIGINAL, "auc_media"]
    c1 = bool(all(auc_bof > resumen.set_index("modelo").loc[b, "auc_media"] for b in BENCHMARKS))
    rechazos = {"P1_nemenyi": bool(f["nemenyi_p"] < ALFA and f["diferencia_media"] > 0),
                "P2_wilcoxon": bool(f["wilcoxon_p_holm"] < ALFA and f["diferencia_media"] > 0),
                "P3_t_corregida": bool(f["t_p_holm"] < ALFA and f["diferencia_media"] > 0),
                "P4_delong": bool(f["delong_p_holm"] < ALFA and f["delong_dif"] > 0)}
    c2 = sum(rechazos.values()) >= 3
    c3 = bool(abs(f["delta_cliff"]) >= 0.147 or abs(f["d_cohen_pareada"]) >= 0.2)
    c4 = bool(adic_pr["bof_mejor_en_media"].sum() >= len(adic_pr) / 2 and not adic_pr["significativamente_peor"].any()) if len(adic_pr) else False
    return {"mejor_benchmark": mejor, "auc_bof": float(auc_bof),
            "auc_mejor_benchmark": float(resumen.set_index("modelo").loc[mejor, "auc_media"]),
            "condicion_1_media_mayor_que_todos": c1, "condicion_2_tres_pruebas": c2, "pruebas_que_rechazan": rechazos,
            "condicion_3_efecto_no_despreciable": c3, "delta_cliff": float(f["delta_cliff"]),
            "d_cohen_pareada": float(f["d_cohen_pareada"]), "condicion_4_consistencia": c4,
            "conjuntos_donde_bof_gana": int(adic_pr["bof_mejor_en_media"].sum()) if len(adic_pr) else 0,
            "conjuntos_significativamente_peor": int(adic_pr["significativamente_peor"].sum()) if len(adic_pr) else 0,
            "superior": bool(c1 and c2 and c3 and c4)}


def _guardar_json(objeto, nombre):
    (TABLAS / nombre).write_text(json.dumps(objeto, indent=1, ensure_ascii=False), encoding="utf-8")


def correr_todo():
    TABLAS.mkdir(parents=True, exist_ok=True)
    filas, pred = cargar()
    resumen = resumen_modelos(filas, pred)
    resumen.to_csv(TABLAS / "resumen_modelos.csv", index=False)
    p, fr, nem = pruebas_bof(filas, pred)
    p.to_csv(TABLAS / "pruebas_bof.csv", index=False)
    nem.to_csv(TABLAS / "nemenyi_p.csv")
    fr["rangos"].rename("rango_medio").to_csv(TABLAS / "rangos_friedman.csv")
    _guardar_json({k: v for k, v in fr.items() if k != "rangos"}, "friedman.json")
    delong_por_repeticion(pred).to_csv(TABLAS / "delong_repeticiones.csv", index=False)
    mejor = resumen[resumen["modelo"] != NOMBRE_ORIGINAL].iloc[0]["modelo"]
    boot = bootstrap_frente_al_mejor(pred, mejor)
    _guardar_json(boot, "bootstrap_bca_mejor.json")
    tabla_pruebas_formal(p, fr, mejor, boot).to_csv(TABLAS / "tabla_pruebas.csv", index=False)
    adic_tabla, adic_pr = conjuntos_adicionales(filas, pred)
    adic_tabla.to_csv(TABLAS / "adicionales_auc.csv", index=False)
    adic_pr.to_csv(TABLAS / "adicionales_pruebas.csv", index=False)
    if not adic_tabla.empty:
        tab, fr2, pares, bs = entre_conjuntos(filas, adic_tabla)
        tab.to_csv(TABLAS / "entre_conjuntos_auc.csv")
        pares.to_csv(TABLAS / "entre_conjuntos_pares.csv", index=False)
        _guardar_json({"friedman": {k: v for k, v in fr2.items() if k != "rangos"}, "rangos": fr2["rangos"].to_dict(),
                       "bayes_vs_mejor_por_conjunto": bs}, "entre_conjuntos.json")
    cal, curvas = calibracion(pred)
    cal.to_csv(TABLAS / "calibracion.csv", index=False)
    curvas.to_csv(TABLAS / "calibracion_curvas.csv", index=False)
    err, por_nota, casos = errores(pred)
    err.to_csv(TABLAS / "errores_subgrupos.csv", index=False)
    por_nota.to_csv(TABLAS / "errores_por_nota.csv")
    casos.to_csv(TABLAS / "errores_casos.csv", index=False)
    if (RESULTADOS / "ablacion_filas.csv").exists():
        ablacion(filas).to_csv(TABLAS / "ablacion.csv", index=False)
    v = veredicto(resumen, p, fr, adic_pr)
    _guardar_json(v, "veredicto.json")
    return v


def robustez():
    """Agrega los estudios de robustez en tablas por estudio y un resumen por modelo."""
    f = pd.read_csv(RESULTADOS / "robustez_filas.csv", keep_default_na=False, na_values=[""])
    f["valor"] = f["valor"].replace({"None": "todas"})
    salida = {}
    ref = f[f["estudio"] == "referencia"]
    salida["robustez_referencia.csv"] = ref[["modelo", "fold", "auc", "segundos"]]
    base = ref.groupby("modelo")["auc"].mean()
    sens = f[f["estudio"] == "sensibilidad"].groupby(["parametro", "valor"], sort=False)["auc"].agg(["mean", "std"]).reset_index()
    salida["robustez_sensibilidad.csv"] = sens.rename(columns={"mean": "auc", "std": "auc_sd"})
    sem = f[f["estudio"] == "semillas"].groupby(["modelo", "valor"])["auc"].mean().reset_index()
    salida["robustez_semillas.csv"] = sem
    ruido = f[f["estudio"] == "ruido"].groupby(["modelo", "parametro", "valor"])["auc"].mean().reset_index()
    ruido["valor"] = ruido["valor"].astype(float)
    ruido["caida"] = ruido["modelo"].map(base) - ruido["auc"]
    salida["robustez_ruido.csv"] = ruido
    tam = f[f["estudio"] == "tamano"].groupby(["modelo", "valor"])["auc"].mean().reset_index()
    tam = pd.concat([tam, base.rename("auc").reset_index().assign(valor="1.0")], ignore_index=True)
    salida["robustez_tamano.csv"] = tam
    conf = f[f["estudio"] == "conforme"].copy()
    conf["cubiertos"] = conf["cobertura"] * conf["n"]
    conf["tam_total"] = conf["tamano_medio"] * conf["n"]
    agg = conf.groupby(["modelo", "alfa", "tipo"]).agg(n=("n", "sum"), cubiertos=("cubiertos", "sum"), tam=("tam_total", "sum")).reset_index()
    agg["cobertura"] = agg["cubiertos"] / agg["n"]
    agg["tamano_medio"] = agg["tam"] / agg["n"]
    por_clase = conf.groupby(["modelo", "alfa", "tipo", "clase"]).agg(n=("n", "sum"), cubiertos=("cubiertos", "sum")).reset_index()
    por_clase["cobertura"] = por_clase["cubiertos"] / por_clase["n"]
    salida["robustez_conforme.csv"] = agg[["modelo", "alfa", "tipo", "cobertura", "tamano_medio"]]
    salida["robustez_conforme_clase.csv"] = por_clase[["modelo", "alfa", "tipo", "clase", "cobertura"]]
    cambio = pd.read_csv(RESULTADOS / "robustez_cambio.csv")
    salida["robustez_cambio.csv"] = cambio
    resumen = pd.DataFrame({"AUC de referencia": base})
    resumen["DE entre 10 semillas"] = sem.groupby("modelo")["auc"].std()
    for (tipo, nivel), col in [(("variables", 0.25), "Caída con ruido σ = 0.25"), (("etiqueta", 0.10), "Caída con 10 % de etiquetas movidas"),
                               (("atipicos", 0.05), "Caída con 5 % de atípicos")]:
        resumen[col] = ruido[(ruido["parametro"] == tipo) & (ruido["valor"] == nivel)].set_index("modelo")["caida"]
    resumen["AUC con 25 % de los datos"] = tam[tam["valor"].astype(str) == "0.25"].set_index("modelo")["auc"]
    resumen["AUC tinto → blanco"] = cambio[cambio["entrena"] == "red"].set_index("modelo")["auc"]
    resumen["AUC blanco → tinto"] = cambio[cambio["entrena"] == "white"].set_index("modelo")["auc"]
    m = agg[(agg["alfa"] == 0.10) & (agg["tipo"] == "marginal")].set_index("modelo")
    resumen["Cobertura conforme 90 %"] = m["cobertura"]
    resumen["Tamaño medio del conjunto"] = m["tamano_medio"]
    if (RESULTADOS / "robustez_bloques.csv").exists():
        bl = pd.read_csv(RESULTADOS / "robustez_bloques.csv")
        salida["robustez_bloques.csv"] = bl
        resumen["AUC con bloques del archivo"] = bl.groupby("modelo")["auc"].mean()
    resumen = resumen.sort_values("AUC de referencia", ascending=False)
    resumen.index.name = "Modelo"
    salida["robustez_resumen.csv"] = resumen.reset_index()
    for nombre, t in salida.items():
        t.to_csv(TABLAS / nombre, index=False)
    return salida
