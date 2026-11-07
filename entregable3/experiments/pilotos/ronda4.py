import sys, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from joblib import Parallel, delayed

sys.path.insert(0, r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\entregable3")
from src.original_model import BosqueOrdinalFocalizado

import importlib.util
spec = importlib.util.spec_from_file_location("dp", r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\modelado_entregable2\modelado\src\data_prep.py",
                                              submodule_search_locations=None)

df = pd.concat([pd.read_csv(r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\modelado_entregable2\modelado\data\winequality-red.csv", sep=";").assign(tipo=0),
                pd.read_csv(r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\modelado_entregable2\modelado\data\winequality-white.csv", sep=";").assign(tipo=1)]).drop_duplicates().reset_index(drop=True)
q = df.pop("quality").to_numpy(int)
X = df.to_numpy(float)
yb = (q >= 7).astype(int)
outer = StratifiedKFold(5, shuffle=True, random_state=42)
tr0, _ = next(outer.split(X, yb))
Xd, qd, yd = X[tr0], q[tr0], yb[tr0]
inner = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=7).split(Xd, yd))

t = time.time()
m = BosqueOrdinalFocalizado(umbral=7, n_estimators=400, ancho_agregacion=0.7 / 0.88, random_state=0).fit(Xd[inner[0][0]], qd[inner[0][0]])
print(f"fit 400 árboles: {time.time() - t:.1f}s  nodos={m.atributo_.size}  oob_auc={m.oob_score_:.4f}  calib={m.calibracion_}")
t = time.time(); p = m.predict_proba(Xd[inner[0][1]])[:, 1]; print(f"predict: {time.time() - t:.2f}s  auc={roc_auc_score(yd[inner[0][1]], p):.4f}")
print("curva oob:", [(a, round(b, 4)) for a, b in m.curva_oob_][::3])


def correr(conf, i, tr, va):
    hs, hp = conf
    mod = BosqueOrdinalFocalizado(umbral=7, n_estimators=400, ancho_particion=hs, ancho_agregacion=hp, random_state=0)
    mod.fit(Xd[tr], qd[tr])
    pr = mod.predict_proba(Xd[va])[:, 1]
    return conf, i, roc_auc_score(yd[va], pr), brier_score_loss(yd[va], pr)


confs = [(0.0, 0.0), (np.inf, 0.0), (np.inf, 0.8), (np.inf, np.inf), (0.8, 0.8)]
res = Parallel(n_jobs=11)(delayed(correr)(c, i, tr, va) for c in confs for i, (tr, va) in enumerate(inner))
r = pd.DataFrame([(str(c), i, a, b) for c, i, a, b in res], columns=["conf", "fold", "auc", "brier"])
print(r.groupby("conf", sort=False)[["auc", "brier"]].mean().round(4))
