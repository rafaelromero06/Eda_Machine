
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from .config import RANDOM_STATE

NOMBRES_BALANCEO = ["sin_balanceo", "SMOTE", "ADASYN", "class_weight"]


class BalancedGaussianNB(GaussianNB):
    """GaussianNB con sample_weight='balanced' calculado en cada fit()."""

    def fit(self, X, y, sample_weight=None):
        pesos = compute_sample_weight(class_weight="balanced", y=y)
        if sample_weight is not None:
            pesos = pesos * sample_weight
        return super().fit(X, y, sample_weight=pesos)


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier con scale_pos_weight = n_neg/n_pos calculado en cada
    fit(), que es el mecanismo idiomático de XGBoost para clases
    desbalanceadas (no tiene class_weight nativo)."""

    def fit(self, X, y, **kwargs):
        y_arr = y.values if hasattr(y, "values") else y
        n_pos = (y_arr == 1).sum()
        n_neg = (y_arr == 0).sum()
        ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
        self.set_params(scale_pos_weight=ratio)
        return super().fit(X, y, **kwargs)


def modelos_que_soportan_class_weight() -> set:
    return {"Regresión Logística (L1/L2)", "Decision Tree", "Random Forest", "SVM"}


def aplicar_balanceo(nombre_balanceo: str, nombre_modelo: str, estimador):
    """Devuelve para insertar
    en el Pipeline de imblearn.

    - estimador_ajustado: el estimador (posiblemente reemplazado/reconfigurado)
    - pasos_muestreo: lista [(nombre, transformador_imblearn)] a insertar
      ANTES del modelo en el Pipeline (vacía si la técnica no es de
      remuestreo)
    - aplica: False si esta combinación (modelo, balanceo) no tiene sentido
      (p. ej. KNN + class_weight) -- el llamador debe omitir esa fila de la
      grilla de experimentos en ese caso.
    """
    if nombre_balanceo == "sin_balanceo":
        return estimador, [], True

    if nombre_balanceo == "SMOTE":
        return estimador, [("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))], True

    if nombre_balanceo == "ADASYN":
        return estimador, [("adasyn", ADASYN(random_state=RANDOM_STATE, n_neighbors=5))], True

    if nombre_balanceo == "class_weight":
        if nombre_modelo in modelos_que_soportan_class_weight():
            estimador = estimador.set_params(class_weight="balanced")
            return estimador, [], True
        if nombre_modelo == "Naive Bayes":
            return BalancedGaussianNB(), [], True
        if nombre_modelo == "XGBoost":
            params = estimador.get_params()
            return BalancedXGBClassifier(**params), [], True
        if nombre_modelo == "KNN":
            return estimador, [], False  # no aplica: sin mecanismo de ponderación
        raise ValueError(f"Modelo sin regla de class_weight definida: {nombre_modelo}")

    raise ValueError(f"Técnica de balanceo desconocida: {nombre_balanceo!r}")
