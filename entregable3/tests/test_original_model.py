import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.original_model import BosqueOrdinalFocalizado, ajustar_platt, auc_mann_whitney, nucleo_umbrales


def datos_ordinales(n=700, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, 6))
    eta = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    latente = eta + rng.logistic(size=n)
    y = np.digitize(latente, [-2.0, -0.7, 0.5, 1.6, 2.8]) + 3
    return X, y


@pytest.fixture(scope="module")
def datos():
    return datos_ordinales()


@pytest.fixture(scope="module")
def modelo(datos):
    X, y = datos
    return BosqueOrdinalFocalizado(umbral=7, n_estimators=60, random_state=0).fit(X[:500], y[:500])


def test_aprende_mejor_que_azar(datos, modelo):
    X, y = datos
    auc = auc_mann_whitney(y[500:] >= 7, modelo.predict_proba(X[500:])[:, 1])
    assert auc > 0.8
    Xb, yb = make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)
    binario = BosqueOrdinalFocalizado(n_estimators=50, random_state=0).fit(Xb, yb)
    assert (binario.predict(Xb) == yb).mean() > 0.7


def test_reproducible_con_misma_semilla(datos):
    X, y = datos
    p1 = BosqueOrdinalFocalizado(umbral=7, n_estimators=30, random_state=1).fit(X, y).predict_proba(X)
    p2 = BosqueOrdinalFocalizado(umbral=7, n_estimators=30, random_state=1).fit(X, y).predict_proba(X)
    p3 = BosqueOrdinalFocalizado(umbral=7, n_estimators=30, random_state=2).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_clone_conserva_hiperparametros(modelo):
    original = BosqueOrdinalFocalizado(umbral=7, n_estimators=10, ancho_agregacion=1.3, desplazamiento=-0.2)
    copia = clone(original)
    assert copia.get_params() == original.get_params()
    copia_ajustado = clone(modelo)
    assert not hasattr(copia_ajustado, "raices_")
    assert copia_ajustado.get_params() == modelo.get_params()


@pytest.mark.parametrize("cambio", [
    {"n_estimators": 0}, {"min_samples_leaf": 0}, {"n_cortes": 0}, {"ancho_agregacion": -1.0},
    {"ancho_particion": float("nan")}, {"max_features": "x"}, {"max_features": 0}, {"max_samples": 1.5},
    {"max_depth": 0}, {"desplazamiento": float("inf")}, {"umbral": 20}, {"umbral": None},
])
def test_hiperparametros_invalidos(datos, cambio):
    X, y = datos
    with pytest.raises(ValueError):
        BosqueOrdinalFocalizado(**{"umbral": 7, "n_estimators": 5, **cambio}).fit(X, y)


def test_entrada_invalida(datos, modelo):
    X, y = datos
    X_malo = X.copy()
    X_malo[0, 0] = np.nan
    with pytest.raises(ValueError):
        BosqueOrdinalFocalizado(umbral=7).fit(X_malo, y)
    X_inf = X.copy()
    X_inf[3, 2] = np.inf
    with pytest.raises(ValueError):
        BosqueOrdinalFocalizado(umbral=7).fit(X_inf, y)
    with pytest.raises(ValueError):
        BosqueOrdinalFocalizado(umbral=7).fit(X, y[:-1])
    with pytest.raises(ValueError):
        BosqueOrdinalFocalizado(umbral=7).fit(X, np.full(len(y), 5))
    with pytest.raises(ValueError):
        modelo.predict_proba(X[:, :4])
    with pytest.raises(ValueError):
        modelo.predict_proba(X_malo)
    with pytest.raises(NotFittedError):
        BosqueOrdinalFocalizado(umbral=7).predict_proba(X)


def test_ganancia_de_cada_division_es_no_negativa(modelo):
    w = modelo.pesos_particion_
    F, peso = modelo.valor_, modelo.peso_nodo_
    G = (F * (1 - F)) @ w
    internos = np.flatnonzero(modelo.atributo_ >= 0)
    izq, der = modelo.izquierdo_[internos], modelo.derecho_[internos]
    ganancia = peso[internos] * G[internos] - peso[izq] * G[izq] - peso[der] * G[der]
    assert internos.size > 0
    assert ganancia.min() >= -1e-9
    np.testing.assert_allclose(peso[internos], peso[izq] + peso[der])


def test_impureza_es_el_minimo_del_brier_ponderado():
    rng = np.random.default_rng(3)
    Z = (rng.integers(3, 9, size=40)[:, None] >= np.arange(4, 9)[None, :]).astype(float)
    w = nucleo_umbrales(np.arange(4, 9), 6.5, 0.9, 1.0)
    F = Z.mean(axis=0)
    impureza = float((F * (1 - F)) @ w)
    brier = lambda u: float(np.mean(((u[None, :] - Z) ** 2) @ w))
    assert brier(F) == pytest.approx(impureza)
    resultado = minimize(brier, x0=np.full(5, 0.5), method="L-BFGS-B", bounds=[(0, 1)] * 5)
    assert resultado.fun == pytest.approx(impureza, abs=1e-8)
    for _ in range(20):
        assert brier(rng.random(5)) >= impureza - 1e-12


def test_terminacion_y_estructura_de_los_arboles(modelo):
    hojas = modelo.atributo_ < 0
    assert hojas.sum() == (~hojas).sum() + modelo.raices_.size
    assert np.all(modelo.filas_nodo_[hojas] >= modelo.min_samples_leaf)
    assert np.all(modelo.izquierdo_[hojas] == -1)


def test_con_ancho_cero_coincide_con_el_bosque_binario(datos):
    X, y = datos
    comun = dict(n_estimators=25, ancho_particion=0.0, ancho_agregacion=0.0, random_state=4)
    ordinal = BosqueOrdinalFocalizado(umbral=7, **comun).fit(X, y)
    binario = BosqueOrdinalFocalizado(**comun).fit(X, (y >= 7).astype(int))
    np.testing.assert_allclose(ordinal.decision_function(X), binario.decision_function(X))


def test_distribucion_acumulada_es_monotona(datos, modelo):
    X, _ = datos
    acum = modelo.predict_distribucion(X[:50])
    assert np.all(acum >= 0) and np.all(acum <= 1)
    assert np.all(np.diff(acum, axis=1) <= 1e-12)
    _, prob = modelo.predict_niveles(X[:50])
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)


def test_calibracion_preserva_el_orden(datos, modelo):
    X, y = datos
    s = modelo.decision_function(X[500:])
    p = modelo.predict_proba(X[500:])[:, 1]
    orden = np.argsort(s, kind="stable")
    assert np.all(np.diff(p[orden]) >= -1e-15)
    assert auc_mann_whitney(y[500:] >= 7, s) == pytest.approx(auc_mann_whitney(y[500:] >= 7, p))


def test_platt_desciende_y_llega_al_optimo():
    rng = np.random.default_rng(5)
    z = rng.normal(size=400)
    y = (rng.random(400) < expit(-0.7 + 1.8 * z)).astype(float)
    a, b, historia = ajustar_platt(z, y)
    assert np.all(np.diff(historia) <= 1e-12)
    nll = lambda th: float(np.sum(np.logaddexp(0, th[0] + th[1] * z) - y * (th[0] + th[1] * z)))
    ref = minimize(nll, x0=[0.0, 0.0], method="BFGS")
    np.testing.assert_allclose([a, b], ref.x, atol=1e-4)


def test_agregacion_conserva_el_orden_bajo_modelo_latente():
    theta = np.array([-2.0, -0.7, 0.5, 1.6, 2.8])
    eta = np.linspace(-4, 4, 400)
    p = expit(eta[:, None] - theta[None, :])
    rng = np.random.default_rng(6)
    for _ in range(10):
        omega = rng.random(theta.size)
        omega /= omega.sum()
        s = p @ omega
        assert np.all(np.diff(s) > 0)
        assert np.array_equal(np.argsort(s), np.argsort(p[:, 3]))


def test_nucleo_umbrales():
    t = np.arange(4, 10, dtype=float)
    w = nucleo_umbrales(t, 7.0, 0.8, 0.9)
    assert w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    assert w[3] == w.max()
    np.testing.assert_allclose(nucleo_umbrales(t, 7.0, 0.0, 1.0), (t == 7).astype(float))
    np.testing.assert_allclose(nucleo_umbrales(t, 7.0, np.inf, 1.0), np.full(6, 1 / 6))
    np.testing.assert_allclose(w[2], w[4])


def test_compatible_con_pipeline_y_validacion_cruzada(datos):
    X, y = datos
    pipe = Pipeline([("esc", StandardScaler()), ("bof", BosqueOrdinalFocalizado(umbral=7, n_estimators=20, random_state=0))])
    puntaje = lambda est, Xv, yv: auc_mann_whitney(np.asarray(yv) >= 7, est.predict_proba(Xv)[:, 1])
    scores = cross_val_score(pipe, X, y, cv=3, scoring=puntaje)
    assert np.all(scores > 0.7)
