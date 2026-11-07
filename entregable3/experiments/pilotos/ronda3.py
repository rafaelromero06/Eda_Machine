import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
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
    if np.isinf(h):
        w = np.ones(len(niveles))
    elif h == 0:
        w = (niveles == 7).astype(float)
    else:
        w = np.exp(-0.5 * ((niveles - centro) / h) ** 2)
    return w / w.sum()


def correr(conf, fold, tr, va):
    boot, ms, leaf, hp, centro = conf
    Z = (qd[tr][:, None] >= niveles[None, :]).astype(float)
    m = ExtraTreesRegressor(400, max_features="sqrt", min_samples_leaf=leaf, bootstrap=boot,
                            max_samples=ms if boot else None, random_state=0, n_jobs=1).fit(Xd[tr], Z)
    P = m.predict(Xd[va])
    s = P @ kernel(hp, centro)
    return conf, fold, roc_auc_score(yd[va], s)


confs = [(False, None, 1, 0.7, 6.5), (True, 1.0, 1, 0.7, 6.5), (True, 0.8, 1, 0.7, 6.5), (True, 0.5, 1, 0.7, 6.5),
         (False, None, 2, 0.7, 6.5), (False, None, 4, 0.7, 6.5), (False, None, 1, 0.4, 6.5), (False, None, 1, 1.0, 6.5),
         (False, None, 1, 0.7, 6.0), (False, None, 1, 0.7, 7.0), (False, None, 1, 0, 6.5), (False, None, 1, np.inf, 6.5)]
res = Parallel(n_jobs=11)(delayed(correr)(c, i, tr, va) for c in confs for i, (tr, va) in enumerate(inner))
r = pd.DataFrame([(str(c), f, a) for c, f, a in res], columns=["conf", "fold", "auc"])
base = r[r.conf == str(confs[0])].set_index("fold")["auc"]
r["dif"] = r.auc - r.fold.map(base)
print(r.groupby("conf", sort=False).agg(auc=("auc", "mean"), dif=("dif", "mean"), pos=("dif", lambda d: int((d > 0).sum()))).round(4).to_string())
