import sys, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestClassifier
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


def kernel(h, centro=6.5):
    if h == 0:
        w = (niveles == 7).astype(float)
    elif np.isinf(h):
        w = np.ones(len(niveles))
    else:
        w = np.exp(-0.5 * ((niveles - centro) / h) ** 2)
    return w / w.sum()


def correr(hs, hp, mf, leaf, fold, tr, va):
    w = kernel(hs); om = kernel(hp)
    Z = (qd[tr][:, None] >= niveles[None, :]).astype(float) * np.sqrt(w)[None, :]
    keep = w > 0
    m = ExtraTreesRegressor(400, max_features=mf, min_samples_leaf=leaf, random_state=0, n_jobs=1).fit(Xd[tr], Z[:, keep])
    P = np.zeros((len(va), len(niveles)))
    pr = m.predict(Xd[va]).reshape(len(va), -1)
    P[:, keep] = pr / np.sqrt(w[keep])[None, :]
    s = (P * om[None, :]).sum(1)
    return hs, hp, mf, leaf, fold, roc_auc_score(yd[va], s)


tareas = []
for mf in ["sqrt", 0.6, 1.0]:
    for hs in [0, 0.7, 1.5, np.inf]:
        for hp in [0, 0.7, 1.5, np.inf]:
            if hs == 0 and hp != 0:
                continue
            for i, (tr, va) in enumerate(inner):
                tareas.append((hs, hp, mf, 1, i, tr, va))
res = Parallel(n_jobs=11)(delayed(correr)(*t) for t in tareas)
r = pd.DataFrame(res, columns=["h_split", "h_score", "max_features", "leaf", "fold", "auc"])
r["max_features"] = r["max_features"].astype(str)
base = r[(r.h_split == 0) & (r.h_score == 0)].set_index(["max_features", "fold"])["auc"]
r["dif"] = r.apply(lambda f: f.auc - base[(f.max_features, f.fold)], axis=1)
t = r.groupby(["max_features", "h_split", "h_score"]).agg(auc=("auc", "mean"), dif=("dif", "mean"), pos=("dif", lambda d: int((d > 0).sum()))).reset_index()
print(t.round(4).to_string(index=False))
