import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from joblib import Parallel, delayed
sys.path.insert(0, r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\entregable3")
from src.original_model import BosqueOrdinalFocalizado
d = r"C:\Users\USER\OneDrive\Escritorio\Eda_Machine\modelado_entregable2\modelado\data"
df = pd.concat([pd.read_csv(d + r"\winequality-red.csv", sep=";").assign(tipo=0), pd.read_csv(d + r"\winequality-white.csv", sep=";").assign(tipo=1)]).drop_duplicates().reset_index(drop=True)
q = df.pop("quality").to_numpy(int); X = df.to_numpy(float); yb = (q >= 7).astype(int)
tr0, _ = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, yb))
Xd, qd, yd = X[tr0], q[tr0], yb[tr0]
inner = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=7).split(Xd, yd))
base = dict(umbral=7, n_estimators=300, ancho_particion=np.inf, ancho_agregacion=0.8, random_state=0)
variantes = {"base": {}, "mf=log2": {"max_features": "log2"}, "mf=5": {"max_features": 5}, "mf=0.6": {"max_features": 0.6},
             "cortes=2": {"n_cortes": 2}, "cortes=4": {"n_cortes": 4}, "hoja=2": {"min_samples_leaf": 2}, "hoja=3": {"min_samples_leaf": 3},
             "desp=-0.5": {"desplazamiento": -0.5}, "desp=+0.5": {"desplazamiento": 0.5}, "agr=0.3": {"ancho_agregacion": 0.3},
             "agr=1.5": {"ancho_agregacion": 1.5}, "part=0.8": {"ancho_particion": 0.8}, "part=2": {"ancho_particion": 2.0}}
def correr(nombre, i, tr, va):
    m = BosqueOrdinalFocalizado(**{**base, **variantes[nombre]}).fit(Xd[tr], qd[tr])
    p = m.predict_proba(Xd[va])[:, 1]
    return nombre, i, roc_auc_score(yd[va], p), brier_score_loss(yd[va], p)
res = Parallel(n_jobs=11)(delayed(correr)(n, i, tr, va) for n in variantes for i, (tr, va) in enumerate(inner))
r = pd.DataFrame(res, columns=["var", "fold", "auc", "brier"])
b = r[r["var"] == "base"].set_index("fold")["auc"]
r["dif"] = r["auc"] - r["fold"].map(b)
print(r.groupby("var", sort=False).agg(auc=("auc", "mean"), dif=("dif", "mean"), pos=("dif", lambda x: int((x > 0).sum())), brier=("brier", "mean")).round(4))
