import sys, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

BASE = Path(r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\modelado_entregable2\modelado")
sys.path.insert(0, str(BASE))
from src.data_prep import preparar_dataset, separar_X_y

df = preparar_dataset(verbose=False)
X, y_reg, y_clf = separar_X_y(df)
X = X.copy(); X["tipo_vino"] = (X["tipo_vino"] == "white").astype(float)
X = X.to_numpy(float); q = y_reg.to_numpy(int); yb = y_clf.to_numpy(int)

outer = StratifiedKFold(5, shuffle=True, random_state=42)
tr0, te0 = next(outer.split(X, yb))
Xd, qd, yd = X[tr0], q[tr0], yb[tr0]
inner = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=7).split(Xd, yd))
niveles = np.arange(4, 10)


def pesos(h, centro=6.5):
    if h == 0:
        w = (niveles == 7).astype(float)
    else:
        w = np.exp(-0.5 * ((niveles - centro) / h) ** 2)
    return w / w.sum()


def objetivos(qq, w):
    Z = (qq[:, None] >= niveles[None, :]).astype(float)
    return Z * np.sqrt(w)[None, :]


def correr(nombre, fold, tr, va):
    Xtr, Xva = Xd[tr], Xd[va]
    t = time.time()
    if nombre == "RFC":
        m = RandomForestClassifier(500, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=1).fit(Xtr, yd[tr])
        s = m.predict_proba(Xva)[:, 1]
    elif nombre == "RFR_q":
        m = RandomForestRegressor(500, max_features="sqrt", random_state=0, n_jobs=1).fit(Xtr, qd[tr])
        s = m.predict(Xva)
    elif nombre.startswith("RFR_foc"):
        h = float(nombre.split("_")[-1]); w = pesos(h)
        m = RandomForestRegressor(500, max_features="sqrt", random_state=0, n_jobs=1).fit(Xtr, objetivos(qd[tr], w))
        s = m.predict(Xva)[:, 3] / np.sqrt(w[3])
    elif nombre == "ETC":
        m = ExtraTreesClassifier(500, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=1).fit(Xtr, yd[tr])
        s = m.predict_proba(Xva)[:, 1]
    elif nombre == "ETR_q":
        m = ExtraTreesRegressor(500, max_features="sqrt", random_state=0, n_jobs=1).fit(Xtr, qd[tr])
        s = m.predict(Xva)
    elif nombre.startswith("ETR_foc"):
        h = float(nombre.split("_")[-1]); w = pesos(h)
        m = ExtraTreesRegressor(500, max_features="sqrt", random_state=0, n_jobs=1).fit(Xtr, objetivos(qd[tr], w))
        s = m.predict(Xva)[:, 3] / np.sqrt(w[3])
    return nombre, fold, roc_auc_score(yd[va], s), time.time() - t


nombres = ["RFC", "RFR_q", "RFR_foc_0.5", "RFR_foc_1.0", "RFR_foc_2.0", "ETC", "ETR_q", "ETR_foc_0.5", "ETR_foc_1.0", "ETR_foc_2.0"]
res = Parallel(n_jobs=11)(delayed(correr)(n, i, tr, va) for n in nombres for i, (tr, va) in enumerate(inner))
r = pd.DataFrame(res, columns=["modelo", "fold", "auc", "seg"])
piv = r.pivot(index="fold", columns="modelo", values="auc")
print(r.groupby("modelo").agg(auc=("auc", "mean"), sd=("auc", "std"), seg=("seg", "mean")).sort_values("auc", ascending=False).round(4))
print("diferencias pareadas vs RFC:")
for c in piv.columns:
    d = piv[c] - piv["RFC"]
    print(f"  {c:12s} {d.mean():+.4f}  (+ en {int((d > 0).sum())}/9)")
