
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .config import RANDOM_STATE

# =============================================================================
# ESPACIOS DE BÚSQUEDA -- CLASIFICACIÓN
# =============================================================================
ESPACIOS_CLASIFICACION = {
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "space": {
            # sqrt(5320) ~= 73; se explora un rango amplio por debajo de eso
            "n_neighbors": {"type": "int", "low": 3, "high": 41, "log": False},
            "weights": {"type": "categorical", "categories": ["uniform", "distance"]},
            "p": {"type": "categorical", "categories": [1, 2]},  # Manhattan vs Euclídea
        },
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "space": {
            # var_smoothing es un factor multiplicativo de la varianza más
            # grande observada; rango log típico recomendado por sklearn docs
            "var_smoothing": {"type": "float", "low": 1e-11, "high": 1e-5, "log": True},
        },
    },
    "Regresión Logística (L1/L2)": {
        # solver="liblinear" soporta l1 y l2 con un solo solver (necesario
        # para poder tratar "penalty" como un hiperparámetro buscable)
        "estimator": LogisticRegression(max_iter=5000, solver="liblinear",
                                         random_state=RANDOM_STATE),
        "space": {
            "C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
            "penalty": {"type": "categorical", "categories": ["l1", "l2"]},
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "space": {
            "max_depth": {"type": "int", "low": 2, "high": 20, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 60, "log": True},
            "criterion": {"type": "categorical", "categories": ["gini", "entropy"]},
        },
    },
    "Random Forest": {
        # n_jobs=1 en el estimador: el paralelismo se maneja a nivel de
        # tarea (ver config.N_WORKERS) para no anidar pools de procesos.
        # n_estimators en [100, 500]: la curva de error de un RF sobre
        # ~5300 filas se aplana bastante antes de 500 árboles, así que el
        # tope cubre la zona útil sin gastar cómputo en árboles que ya no
        # mejoran el modelo. (En la primera corrida, hecha en un entorno de
        # 1 solo núcleo, el tope se bajó a 200 porque RF era ~73% del
        # tiempo total; con varios núcleos se restituye el rango completo.)
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 3, "high": 25, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20, "log": True},
            "max_features": {"type": "categorical", "categories": ["sqrt", "log2", None]},
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                    n_jobs=1, tree_method="hist"),
        "space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 2, "high": 10, "log": False},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        },
    },
    "SVM": {
        # probability=False: decision_function alcanza para ROC-AUC y es
        # mucho más barato. probability=True/CalibratedClassifierCV se usa
        # aparte, solo para el análisis de calibración ,
        # donde precisamente se espera que SVM salga mal calibrado.
        "estimator": SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
        "space": {
            "C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "gamma": {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
        },
    },
}

# =============================================================================
# ESPACIOS DE BÚSQUEDA -- REGRESIÓN
# =============================================================================
ESPACIOS_REGRESION = {
    "KNN": {
        "estimator": KNeighborsRegressor(),
        "space": {
            "n_neighbors": {"type": "int", "low": 3, "high": 41, "log": False},
            "weights": {"type": "categorical", "categories": ["uniform", "distance"]},
            "p": {"type": "categorical", "categories": [1, 2]},
        },
    },
    "Ridge": {
        "estimator": Ridge(random_state=RANDOM_STATE),
        "space": {
            "alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
        },
    },
    "Lasso": {
        "estimator": Lasso(random_state=RANDOM_STATE, max_iter=20000),
        "space": {
            "alpha": {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "space": {
            "max_depth": {"type": "int", "low": 2, "high": 20, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 60, "log": True},
        },
    },
    "Random Forest": {
        # Ver justificación del rango de n_estimators en la versión de
        # clasificación de este mismo diccionario, arriba.
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 3, "high": 25, "log": False},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20, "log": True},
            "max_features": {"type": "categorical", "categories": ["sqrt", "log2", None]},
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
        "space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
            "max_depth": {"type": "int", "low": 2, "high": 10, "log": False},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        },
    },
    "SVR": {
        "estimator": SVR(kernel="rbf"),
        "space": {
            "C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True},
            "gamma": {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
            "epsilon": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
        },
    },
}


def obtener_espacios(tarea: str) -> dict:
    """tarea: 'clasificacion' | 'regresion'."""
    if tarea == "clasificacion":
        return ESPACIOS_CLASIFICACION
    elif tarea == "regresion":
        return ESPACIOS_REGRESION
    raise ValueError(f"tarea desconocida: {tarea!r}")


def nuevo_estimador(tarea: str, nombre_modelo: str):
    """Devuelve una copia SIN AJUSTAR del estimador (clone de sklearn) --
    nunca se reutiliza una instancia ya ajustada entre folds."""
    espacio = obtener_espacios(tarea)[nombre_modelo]
    return clone(espacio["estimator"])
