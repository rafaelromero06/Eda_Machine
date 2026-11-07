from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .original_model import BosqueOrdinalFocalizado

SEMILLA = 42
NOMBRE_ORIGINAL = "Bosque Ordinal Focalizado"


class BalancedGaussianNB(GaussianNB):
    """GaussianNB con pesos 'balanced' recalculados en cada ajuste."""

    def fit(self, X, y, sample_weight=None):
        pesos = compute_sample_weight(class_weight="balanced", y=y)
        if sample_weight is not None:
            pesos = pesos * sample_weight
        return super().fit(X, y, sample_weight=pesos)


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier con scale_pos_weight = n_neg / n_pos recalculado en cada ajuste."""

    def fit(self, X, y, **kwargs):
        y_arr = np.asarray(y)
        n_pos = (y_arr == 1).sum()
        self.set_params(scale_pos_weight=((y_arr == 0).sum() / n_pos) if n_pos > 0 else 1.0)
        return super().fit(X, y, **kwargs)


class CatBoostEnvoltorio(ClassifierMixin, BaseEstimator):
    """Interfaz de scikit-learn estable para CatBoost (su get_params omite los valores None)."""

    def __init__(self, iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, bagging_temperature=1.0,
                 ponderacion="ninguna", random_seed=SEMILLA, thread_count=1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.ponderacion = ponderacion
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.modelo_ = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, bagging_temperature=self.bagging_temperature,
            auto_class_weights=None if self.ponderacion == "ninguna" else "Balanced",
            random_seed=self.random_seed, thread_count=self.thread_count, verbose=0, allow_writing_files=False)
        self.modelo_.fit(np.asarray(X), np.asarray(y))
        self.classes_ = np.asarray(self.modelo_.classes_)
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


ESPACIOS = {
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "space": {
            "n_neighbors": {"type": "int", "low": 3, "high": 41, "log": False},
            "weights": {"type": "categorical", "categories": ["uniform", "distance"]},
            "p": {"type": "categorical", "categories": [1, 2]},
        },
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "space": {"var_smoothing": {"type": "float", "low": 1e-11, "high": 1e-5, "log": True}},
    },
    "Regresión Logística (L1/L2)": {
        "estimator": LogisticRegression(max_iter=5000, solver="liblinear", random_state=SEMILLA),
        "space": {
            "C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
            "penalty": {"type": "categorical", "categories": ["l1", "l2"]},
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=SEMILLA),
        "space": {
            "max_depth": {"type": "int", "low": 2, "high": 20, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 60, "log": True},
            "criterion": {"type": "categorical", "categories": ["gini", "entropy"]},
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=SEMILLA, n_jobs=1),
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 3, "high": 25, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20, "log": True},
            "max_features": {"type": "categorical", "categories": ["sqrt", "log2", None]},
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=SEMILLA, eval_metric="logloss", n_jobs=1, tree_method="hist"),
        "space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 2, "high": 10, "log": False},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        },
    },
    "SVM": {
        "estimator": SVC(kernel="rbf", probability=False, random_state=SEMILLA),
        "space": {
            "C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "gamma": {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
        },
    },
    "LightGBM": {
        "estimator": LGBMClassifier(random_state=SEMILLA, n_jobs=1, verbose=-1, subsample_freq=1),
        "space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 8, "high": 128, "log": True},
            "min_child_samples": {"type": "int", "low": 5, "high": 100, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "class_weight": {"type": "categorical", "categories": [None, "balanced"]},
        },
        "inicial": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20,
                    "subsample": 1.0, "colsample_bytree": 1.0, "class_weight": None},
    },
    "CatBoost": {
        "estimator": CatBoostEnvoltorio(),
        "space": {
            "iterations": {"type": "int", "low": 100, "high": 1000, "log": False},
            "learning_rate": {"type": "float", "low": 1e-2, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 4, "high": 10, "log": False},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
            "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
            "ponderacion": {"type": "categorical", "categories": ["ninguna", "balanceada"]},
        },
        "inicial": {"iterations": 1000, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0,
                    "bagging_temperature": 1.0, "ponderacion": "ninguna"},
    },
    NOMBRE_ORIGINAL: {
        "estimator": BosqueOrdinalFocalizado(random_state=SEMILLA),
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "log": False},
            "max_features": {"type": "categorical", "categories": ["sqrt", "log2", None]},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20, "log": True},
            "ancho_agregacion": {"type": "float", "low": 0.1, "high": 5.0, "log": True},
            "ancho_particion": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            "desplazamiento": {"type": "float", "low": -1.0, "high": 1.0, "log": False},
        },
    },
}

MEJORES = {
    "Random Forest": ("class_weight", "Grid Search"),
    "XGBoost": ("SMOTE", "Bayesiano"),
    "SVM": ("class_weight", "Random Search"),
    "KNN": ("sin_balanceo", "Bayesiano"),
    "Regresión Logística (L1/L2)": ("sin_balanceo", "Bayesiano"),
    "Decision Tree": ("sin_balanceo", "Random Search"),
    "Naive Bayes": ("class_weight", "Bayesiano"),
    "LightGBM": ("ponderación en la búsqueda", "Bayesiano"),
    "CatBoost": ("ponderación en la búsqueda", "Bayesiano"),
    NOMBRE_ORIGINAL: ("no aplica (usa la nota ordinal)", "Bayesiano"),
}
BENCHMARKS_CLASE = ["Random Forest", "XGBoost", "SVM", "KNN", "Regresión Logística (L1/L2)", "Decision Tree", "Naive Bayes"]
BENCHMARKS_SOTA = ["LightGBM", "CatBoost"]
MODELOS = BENCHMARKS_CLASE + BENCHMARKS_SOTA + [NOMBRE_ORIGINAL]
CORTOS = {"Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM", "KNN": "KNN", "Regresión Logística (L1/L2)": "RL",
          "Decision Tree": "DT", "Naive Bayes": "NB", "LightGBM": "LGBM", "CatBoost": "CB", NOMBRE_ORIGINAL: "BOF"}


def construir_preprocesador(numericas, categoricas, imputar=False):
    """Escalado de numéricas y one-hot de categóricas (con imputación por mediana/moda si ``imputar``)."""
    if not imputar:
        return ColumnTransformer([("num", StandardScaler(), list(numericas)),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), list(categoricas))])
    partes = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), list(numericas))]
    if categoricas:
        partes.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("ohe", OneHotEncoder(handle_unknown="ignore"))]), list(categoricas)))
    return ColumnTransformer(partes, sparse_threshold=0.0)


def _balancear(nombre_modelo, balanceo, estimador):
    if balanceo in ("sin_balanceo",) or nombre_modelo in BENCHMARKS_SOTA + [NOMBRE_ORIGINAL]:
        return estimador, []
    if balanceo == "SMOTE":
        return estimador, [("smote", SMOTE(random_state=SEMILLA, k_neighbors=5))]
    if balanceo == "ADASYN":
        return estimador, [("adasyn", ADASYN(random_state=SEMILLA, n_neighbors=5))]
    if balanceo == "class_weight":
        if nombre_modelo in {"Regresión Logística (L1/L2)", "Decision Tree", "Random Forest", "SVM"}:
            return estimador.set_params(class_weight="balanced"), []
        if nombre_modelo == "Naive Bayes":
            return BalancedGaussianNB(), []
        if nombre_modelo == "XGBoost":
            return BalancedXGBClassifier(**estimador.get_params()), []
    raise ValueError(f"Combinación no válida: {nombre_modelo} con {balanceo}")


def construir_pipeline(nombre_modelo, datos, balanceo=None):
    """Pipeline (preprocesamiento + balanceo + modelo) de la mejor combinación de un modelo para un conjunto."""
    balanceo = MEJORES[nombre_modelo][0] if balanceo is None else balanceo
    estimador = clone(ESPACIOS[nombre_modelo]["estimator"])
    if nombre_modelo == NOMBRE_ORIGINAL:
        estimador.set_params(umbral=datos["umbral"])
    estimador, muestreo = _balancear(nombre_modelo, balanceo, estimador)
    prep = construir_preprocesador(datos["numericas"], datos["categoricas"], datos.get("imputar", False))
    return ImbPipeline([("prep", prep)] + muestreo + [("modelo", estimador)])


def usa_respuesta_ordinal(nombre_modelo) -> bool:
    return nombre_modelo == NOMBRE_ORIGINAL


def puntaje(pipe, X):
    """Probabilidad del evento si el modelo la entrega; si no (SVM), la función de decisión."""
    if hasattr(pipe.named_steps["modelo"], "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def da_probabilidades(nombre_modelo) -> bool:
    return nombre_modelo != "SVM"
