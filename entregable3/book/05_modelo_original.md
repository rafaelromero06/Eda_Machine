---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# El modelo original: Bosque Ordinal Focalizado

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib, subprocess
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from myst_nb import glue
from src.libro import configurar, mostrar, tabla, resultado, AZUL, NARANJA, GRIS, TINTA_2, ORIGINAL, CORTOS
configurar()
```

## Motivación: la limitación de binarizar la nota

Todos los modelos del Entregable 2 aprenden con la etiqueta $Y = \mathbb{1}\{\text{nota} \ge 7\}$. Esa etiqueta trata igual a un vino de 6, que estuvo a un punto de ser bueno, y a uno de 3. Binarizar una respuesta ordinal o continua desperdicia información y reduce la potencia estadística {cite:p}`altman2006,harrell2018`. En el problema del vino la pérdida es concreta: 4 de cada 5 vinos son «no buenos» y casi la mitad de ellos tiene nota 6.

La evidencia de que esto limita a los modelos existentes viene de los experimentos piloto ({numref}`fig-pilotos`), hechos solo con validación interna dentro del entrenamiento del pliegue externo 1: un Extra-Trees que aprende la nota completa supera al mismo método entrenado con la etiqueta binaria, y combinar las probabilidades acumuladas de los umbrales vecinos al 7 supera tanto a usar solo el umbral 7 como a darles a todos el mismo peso.

```{code-cell} ipython3
:tags: [hide-input]
b = pd.read_csv(RAIZ / "experiments" / "bitacora_diseno.csv")
r2 = b[(b["ronda"] == 2) & b["configuracion"].str.startswith("mf=sqrt")].copy()
r2["h_part"] = r2["configuracion"].str.extract(r"h_part=([^,]+)")[0]
r2["h_agr"] = r2["configuracion"].str.extract(r"h_agr=(.+)$")[0]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
etiquetas = {"0.0": "solo umbral 7", "0.7": "ancho 0.7", "1.5": "ancho 1.5", "inf": "uniforme"}
for hp, color, marca in [("0.0", GRIS, "o"), ("0.7", AZUL, "o"), ("1.5", "#86b6ef", "s"), ("inf", NARANJA, "D")]:
    d = r2[r2["h_agr"] == hp].set_index("h_part").reindex(["0.0", "0.7", "1.5", "inf"]).dropna(subset=["auc_media"])
    ax.plot(range(len(d)), d["auc_media"], marker=marca, color=color, lw=2, ms=7, label=f"agregación: {etiquetas[hp]}")
ax.set_xticks(range(4), ["binaria", "ancho 0.7", "ancho 1.5", "uniforme"])
ax.set_xlabel("Núcleo de la partición (criterio de los árboles)")
ax.set_ylabel("AUC medio (9 pliegues internos)")
ax.set_title("Pilotos: usar la nota completa y combinar umbrales vecinos sube el AUC")
ax.legend(fontsize=8, ncol=2, loc="lower right")
plt.tight_layout()
glue("fig_pilotos", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_pilotos
:name: fig-pilotos
AUC de Extra-Trees de salida múltiple con raíz de p atributos candidatos en la ronda 2 de diseño. La referencia binaria (partición y agregación binarias) da 0.848; la mejor combinación, 0.854. Los datos completos de las 80 configuraciones de diseño están en `experiments/bitacora_diseno.csv`.
```

## Formulación

Sea $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ con $y_i$ en niveles ordenados $\ell_1 < \dots < \ell_L$, y sea $\tau$ el umbral del evento ($\tau = 7$ en el vino), de modo que $Y_i = \mathbb{1}\{y_i \ge \tau\}$. Los umbrales $t_1 < \dots < t_K$ son los niveles salvo el menor (más $\tau$ si no estuviera), y cada vino se describe por sus indicadores acumulados $Z_{ik} = \mathbb{1}\{y_i \ge t_k\}$, que son no crecientes en $k$.

**Núcleos.** Con $s$ la desviación estándar de $y$ y el centro $c = \tau + \delta s$, los pesos de un núcleo de ancho relativo $h$ son

$$
\omega_k(h) = \frac{\exp\{-(t_k - c)^2 / (2 h^2 s^2)\}}{\sum_{j=1}^{K} \exp\{-(t_j - c)^2 / (2 h^2 s^2)\}},
$$ (eq-nucleo)

con los límites $h \to 0$ (todo el peso en el umbral más cercano a $c$) y $h \to \infty$ (pesos iguales). El modelo usa dos núcleos: $w = \omega(h_p)$ para partir y $\omega = \omega(h_a)$ para puntuar.

**Impureza y ganancia.** En un nodo $S$ con pesos bootstrap $v_i$, $n_S = \sum_{i \in S} v_i$ y $F_k(S) = n_S^{-1} \sum_{i \in S} v_i Z_{ik}$, la impureza acumulada ponderada es

$$
G_w(S) = \sum_{k=1}^{K} w_k\, F_k(S)\,\bigl(1 - F_k(S)\bigr),
$$ (eq-impureza)

y la ganancia de dividir $S$ en $S_L$ y $S_R$ es

$$
\Delta = n_S\, G_w(S) - n_L\, G_w(S_L) - n_R\, G_w(S_R) = \sum_{k} w_k \left( \frac{A_{Lk}^2}{n_L} + \frac{A_{Rk}^2}{n_R} - \frac{A_k^2}{n_S} \right),
$$ (eq-ganancia)

donde $A_k = \sum_{i \in S} v_i Z_{ik}$. La segunda forma es la que se programa: solo necesita sumas por nodo y por candidato.

**Puntaje y calibración.** Con $\hat F_k(\mathbf{x}) = T^{-1} \sum_{b=1}^{T} F_k(\text{hoja}_b(\mathbf{x}))$, el puntaje y la probabilidad son

$$
s(\mathbf{x}) = \sum_{k=1}^{K} \omega_k\, \hat F_k(\mathbf{x}), \qquad \hat p(\mathbf{x}) = \sigma\bigl(a + b\, \operatorname{logit} s(\mathbf{x})\bigr),
$$ (eq-puntaje)

con $(a, b)$ ajustados por máxima verosimilitud sobre los puntajes fuera de bolsa $s_{\text{oob}}(\mathbf{x}_i)$ y las etiquetas $Y_i$:

$$
(a, b) = \arg\min_{a, b} \sum_{i} \Bigl[ \log\bigl(1 + e^{a + b z_i}\bigr) - Y_i (a + b z_i) \Bigr], \qquad z_i = \operatorname{logit} s_{\text{oob}}(\mathbf{x}_i).
$$ (eq-platt)

**Supuestos.** (S1) Los vinos son intercambiables (no hay estructura temporal); (S2) la nota resulta de cortar una calidad latente con umbrales fijos, de modo que las probabilidades acumuladas vecinas comparten información sobre el mismo vino; (S3) la relación entre química y calidad es suave a trozos, lo que los árboles capturan. S2 es el supuesto clásico de los modelos ordinales {cite:p}`mccullagh1980` y es plausible para una nota que es la mediana de catadores en una escala de 0 a 10.

## Pseudocódigo

```{literalinclude} ../src/tablero_original.py
:language: text
:start-after: 'PSEUDOCODIGO = """'
:end-before: '"""'
```

## Sustento teórico

```{prf:proposition}
:label: prop-brier
La impureza {eq}`eq-impureza` es el mínimo del puntaje de Brier ponderado de las probabilidades acumuladas del nodo:
$G_w(S) = \min_{\mathbf{u} \in [0,1]^K} n_S^{-1} \sum_{i \in S} v_i \sum_k w_k (u_k - Z_{ik})^2$, y el mínimo se alcanza en $\mathbf{u} = \mathbf{F}(S)$.
```

*Demostración.* La función es separable en $k$. Para cada $k$, $\phi_k(u) = n_S^{-1} \sum_i v_i (u - Z_{ik})^2$ es una cuadrática convexa con derivada $2(u - F_k)$, que se anula en $u = F_k$. Como $Z_{ik}^2 = Z_{ik}$, su valor mínimo es $n_S^{-1} \sum_i v_i Z_{ik} - F_k^2 = F_k(1 - F_k)$. Al sumar con los pesos $w_k \ge 0$ se obtiene $G_w(S)$. $\square$

El puntaje minimizado es, para respuestas ordinales, la versión ponderada del puntaje de probabilidad por rangos {cite:p}`epstein1969` y la versión discreta del CRPS ponderado por umbral {cite:p}`gneiting2011`. Es una regla de puntuación propia (estrictamente propia si todos los $w_k > 0$) por ser una suma ponderada de puntajes de Brier de eventos {cite:p}`brier1950,gneiting2007`. Cada árbol, por lo tanto, minimiza de forma voraz una regla propia centrada en el umbral que importa.

```{prf:corollary}
:label: cor-ganancia
Para cualquier partición, $\Delta \ge 0$, con igualdad si y solo si $F_k(S_L) = F_k(S_R)$ para todo $k$ con $w_k > 0$. En consecuencia, la impureza total de entrenamiento $\sum_{\text{hojas}} n_h G_w(h)$ no aumenta en ninguna división, y el crecimiento de cada árbol termina tras a lo sumo $n_b - 1$ divisiones, donde $n_b$ es el número de vinos distintos de su muestra bootstrap.
```

*Demostración.* Por la proposición, $n_S G_w(S) = \min_{\mathbf u} \sum_{i \in S} v_i \ell(\mathbf u, Z_i) \ge \min_{\mathbf u} \sum_{S_L} + \min_{\mathbf u} \sum_{S_R} = n_L G_w(S_L) + n_R G_w(S_R)$, porque usar el mismo $\mathbf u$ en ambos hijos es una opción factible. La igualdad exige que $\mathbf F(S)$ minimice en ambos hijos, y por convexidad estricta de cada coordenada con $w_k > 0$ eso ocurre si y solo si $\mathbf F(S_L) = \mathbf F(S_R)$. Cada división crea dos nodos con al menos $\ell \ge 1$ vinos distintos; las hojas forman una partición de los $n_b$ vinos, así que hay a lo sumo $n_b$ hojas y $n_b - 1$ nodos internos. $\square$

```{prf:proposition}
:label: prop-orden
Si $P(y \ge t_k \mid \mathbf x) = \Psi(\eta(\mathbf x) - \theta_k)$ para una distribución continua estrictamente creciente $\Psi$ y umbrales $\theta_1 < \dots < \theta_K$ (modelo latente ordinal), entonces para cualquier vector de pesos $\omega \ge 0$ con $\sum_k \omega_k = 1$ el puntaje poblacional $s^\ast(\mathbf x) = \sum_k \omega_k \Psi(\eta(\mathbf x) - \theta_k)$ es una función estrictamente creciente de $P(Y = 1 \mid \mathbf x)$. Por lo tanto ordena los vinos igual que la probabilidad del evento y su AUC es el máximo posible.
```

*Demostración.* Cada término $\Psi(\eta - \theta_k)$ es estrictamente creciente en $\eta$ y al menos un $\omega_k$ es positivo, así que $s^\ast$ es estrictamente creciente en $\eta$. También lo es $P(Y=1 \mid \mathbf x) = \Psi(\eta - \theta_{k_\tau})$. Componer una función estrictamente creciente con la inversa de otra da una función estrictamente creciente, de modo que $s^\ast = g(P(Y = 1 \mid \mathbf x))$ con $g$ creciente. El AUC es invariante ante transformaciones estrictamente crecientes del puntaje y se maximiza con cualquier transformación creciente de la probabilidad a posteriori {cite:p}`clemencon2008`. $\square$

La proposición dice que combinar umbrales **no cambia el blanco**: en la población, el puntaje combinado y la probabilidad del evento ordenan igual. La ventaja aparece en muestras finitas, porque los umbrales vecinos aportan información sobre el mismo $\eta(\mathbf x)$:

```{prf:proposition}
:label: prop-snr
Sea $\hat{\mathbf F}$ un estimador de las probabilidades acumuladas con covarianza local $C/m$ y sensibilidad $\mathbf g = \partial \mathbf F / \partial \eta$. Entre los puntajes lineales $s = \omega^\top \hat{\mathbf F}$, la razón señal-ruido local $\rho(\omega) = (\omega^\top \mathbf g)^2 / (\omega^\top C \omega)$ se maximiza con $\omega^\ast \propto C^{-1} \mathbf g$, y $\rho(\omega^\ast) \ge \rho(\mathbf e_\tau)$, donde $\mathbf e_\tau$ usa solo el umbral del evento.
```

*Demostración.* Por Cauchy–Schwarz con el producto interno $\langle \mathbf a, \mathbf b \rangle_C = \mathbf a^\top C \mathbf b$: $(\omega^\top \mathbf g)^2 = \langle \omega, C^{-1}\mathbf g \rangle_C^2 \le (\omega^\top C \omega)(\mathbf g^\top C^{-1} \mathbf g)$, con igualdad en $\omega \propto C^{-1} \mathbf g$. El valor máximo $\mathbf g^\top C^{-1} \mathbf g$ es mayor o igual que el de cualquier otro $\omega$, en particular $\mathbf e_\tau$. $\square$

El núcleo gaussiano es una familia de dos parámetros que contiene a $\mathbf e_\tau$ ($h_a = 0$) y a los pesos uniformes ($h_a = \infty$); la búsqueda bayesiana elige el ancho dentro de la validación interna, así que la mejor configuración de la familia es al menos tan buena como la binaria en el criterio de búsqueda.

```{prf:proposition}
:label: prop-platt
El problema {eq}`eq-platt` es convexo; el método de Newton con búsqueda lineal de Armijo produce una log-pérdida no creciente que converge al mínimo global. Si $b > 0$, la calibración es estrictamente creciente en $s$ y no cambia el AUC.
```

*Demostración.* La hessiana es $\sum_i \pi_i(1-\pi_i)(1, z_i)^\top(1, z_i) \succeq 0$, así que el problema es convexo; la condición de Armijo garantiza descenso en cada paso y el método converge al mínimo de una función convexa diferenciable {cite:p}`boyd2004`. La composición $\sigma(a + b\,\operatorname{logit}(s))$ es creciente si $b > 0$, y el AUC solo depende del orden. $\square$

Las pruebas unitarias verifican numéricamente cada propiedad: la ganancia no negativa en todos los nodos de un bosque ajustado, la igualdad entre la impureza y el mínimo del Brier ponderado (contra un optimizador numérico), la reducción exacta al bosque binario con anchos cero, la monotonía de la distribución acumulada, el descenso de la log-pérdida de la calibración y la conservación del orden bajo el modelo latente.

## Complejidad

En cada nivel del árbol, con $R_d \le n_b$ filas en nodos que todavía se dividen, $m$ atributos candidatos, $r$ cortes por atributo y $K$ umbrales, el algoritmo hace: ordenar las filas por nodo y calcular mínimos y máximos por atributo, $O(R_d \log R_d + R_d\, p)$; acumular las sumas de los candidatos con un producto disperso, $O(R_d\, m\, r\, K)$; y elegir el mejor candidato, $O(J_d\, m\, r\, K)$ con $J_d \le R_d$ nodos. Con profundidad $D$ y $T$ árboles, el entrenamiento cuesta

$$
O\bigl(T\, D\, n\, (\log n + p + m\, r\, K)\bigr),
$$ (eq-complejidad)

que para árboles balanceados ($D = O(\log n)$) es $O(T\, n \log n\,(p + m r K))$. La memoria es $O(T\, n\, K)$ (a lo sumo $2 n_b - 1$ nodos por árbol, cada uno con su vector acumulado), y predecir un vino cuesta $O(T\, D)$, porque el puntaje combinado de cada hoja se precalcula. La calibración fuera de bolsa agrega $O(T\, n\, D)$.

```{code-cell} ipython3
:tags: [hide-input]
esc = resultado("robustez_escalabilidad.csv")
bof = esc[esc["modelo"] == ORIGINAL]
n_ = bof[bof["eje"] == "n"].groupby("n")["ajuste_s"].median()
p_ = bof[bof["eje"] == "p"].groupby("p")["ajuste_s"].median()
pend_n = np.polyfit(np.log(n_.index), np.log(n_.values), 1)[0]
pend_p = np.polyfit(np.log(p_.index), np.log(p_.values), 1)[0]
glue("pend_n", f"{pend_n:.2f}", display=False)
glue("pend_p", f"{pend_p:.2f}", display=False)
fig, ejes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, s, nombre, pend in [(ejes[0], n_, "n (vinos de entrenamiento)", pend_n), (ejes[1], p_, "p (variables)", pend_p)]:
    ax.loglog(s.index, s.values, marker="o", color=AZUL, lw=2)
    ax.set_xlabel(nombre + " — escala log")
    ax.set_ylabel("Segundos de ajuste — escala log")
    ax.set_title(f"Pendiente log-log = {pend:.2f}")
plt.tight_layout()
glue("fig_escala", fig, display=False)
plt.close(fig)
```

```{glue:figure} fig_escala
:name: fig-escala
Tiempo de ajuste del Bosque Ordinal Focalizado con sus hiperparámetros finales, en un solo proceso, al variar el número de vinos y el número de variables (se agregan variables de ruido). Cada punto es la mediana de 3 repeticiones, o una sola medición cuando el ajuste con todos los datos tarda más de 30 s.
```

La pendiente empírica en $n$ es {glue:text}`pend_n` y en $p$ es {glue:text}`pend_p`. Ambas son coherentes con {eq}`eq-complejidad`: el costo crece algo más que linealmente con $n$ (por el factor $D$, que aumenta con la muestra en árboles completos) y menos que linealmente con $p$, porque solo los $m$ atributos candidatos pagan el costo de los cortes y $m$ crece como $\sqrt p$.

## Implementación

El modelo está en `src/original_model.py` (unas 450 líneas). Solo usa numpy y scipy para el cálculo; de scikit-learn hereda `BaseEstimator` y `ClassifierMixin`, y usa sus funciones de validación de entradas, como sugiere el esqueleto del curso. La función central construye un lote de árboles nivel por nivel:

```{literalinclude} ../src/original_model.py
:pyobject: BosqueOrdinalFocalizado._crecer_lote
```

Cómo se lee, de arriba abajo:

- `np.nonzero(conteos)` convierte los conteos bootstrap de cada árbol en filas `(árbol, vino, peso)`; al empezar, el nodo de cada fila es la raíz de su árbol, así que todos los árboles del lote crecen a la vez.
- `S = A @ (peso * Z)` suma, para cada nodo, los indicadores acumulados ponderados ($A_k$ de {eq}`eq-ganancia`); `G` es la impureza {eq}`eq-impureza`. Un nodo se puede dividir si tiene al menos `2 * hoja_min` vinos distintos e impureza positiva.
- Para los nodos divisibles, las filas se ordenan por nodo y `reduceat` da el mínimo y el máximo de cada atributo; los atributos constantes reciben una clave alta para no ser elegidos. Se toman `m` atributos al azar y `n_cortes` cortes uniformes entre su mínimo y su máximo (la aleatorización de Extra-Trees).
- `izq` marca, para cada fila y cada candidato, si va a la izquierda. La matriz dispersa `Q` (una fila por nodo y candidato) suma en una sola multiplicación el número de vinos, el peso y los indicadores que van a la izquierda; los de la derecha se obtienen por resta.
- `ganancia` es la segunda forma de {eq}`eq-ganancia`; los candidatos que dejan un hijo con menos de `hoja_min` vinos o usan un atributo constante quedan en $-\infty$. Se elige el mejor por nodo y se divide si la ganancia es positiva.
- Cada nivel guarda atributo, corte, valores acumulados `F` (también de los nodos internos, que SHAP necesita), pesos y el rango de los hijos; las filas de los nodos divididos pasan al siguiente nivel con su nuevo nodo `2 * rango + derecha`.

Después, `_ensamblar` une los niveles en arreglos planos (atributo, corte, hijos y valores de todos los nodos de todos los árboles), `_hojas` recorre todos los árboles a la vez para un conjunto de vinos y `_fuera_de_bolsa` calcula el puntaje con los árboles que no vieron a cada vino, la curva de convergencia y la calibración {eq}`eq-platt`.

## Pruebas unitarias

```{code-cell} ipython3
:tags: [hide-input]
salida = subprocess.run([sys.executable, "-m", "pytest", "tests/test_original_model.py", "-q", "-p", "no:cacheprovider"],
                        cwd=RAIZ, capture_output=True, text=True)
print(salida.stdout.strip().splitlines()[-1])
```

| Prueba | Qué verifica | Requisito del curso |
|---|---|---|
| `test_aprende_mejor_que_azar` | AUC > 0.8 con datos ordinales sintéticos y exactitud > 0.7 con `make_classification` | (i) aprende cuando la solución es conocida |
| `test_reproducible_con_misma_semilla` | Misma semilla, mismas probabilidades; otra semilla, otras | (ii) reproducibilidad |
| `test_entrada_invalida`, `test_hiperparametros_invalidos` (12 casos) | NaN, infinitos, longitudes distintas, una sola clase, umbral fuera de rango e hiperparámetros inválidos lanzan `ValueError`; predecir sin ajustar lanza `NotFittedError` | (iii) entradas inválidas |
| `test_clone_conserva_hiperparametros` | `sklearn.base.clone` copia los hiperparámetros y no los atributos ajustados | (iv) compatibilidad con `clone` |
| `test_ganancia_de_cada_division_es_no_negativa` | $\Delta \ge 0$ en todos los nodos internos y el peso del padre es la suma de los hijos | (v) {prf:ref}`cor-ganancia` |
| `test_impureza_es_el_minimo_del_brier_ponderado` | $G_w(S)$ coincide con el mínimo numérico del Brier ponderado | (v) {prf:ref}`prop-brier` |
| `test_terminacion_y_estructura_de_los_arboles` | hojas = internos + árboles y cada hoja tiene al menos `min_samples_leaf` vinos | (v) {prf:ref}`cor-ganancia` |
| `test_con_ancho_cero_coincide_con_el_bosque_binario` | Con anchos cero el modelo ordinal es idéntico al entrenado con la etiqueta binaria | caso límite de {eq}`eq-nucleo` |
| `test_distribucion_acumulada_es_monotona` | $\hat F_k$ no crece con $k$ y las probabilidades por nivel suman 1 | estructura ordinal |
| `test_calibracion_preserva_el_orden` | La calibración no cambia el AUC | (v) {prf:ref}`prop-platt` |
| `test_platt_desciende_y_llega_al_optimo` | La log-pérdida nunca sube y coincide con BFGS | (v) {prf:ref}`prop-platt` |
| `test_agregacion_conserva_el_orden_bajo_modelo_latente` | El puntaje combinado ordena igual que $P(Y = 1 \mid \mathbf x)$ con pesos al azar | (v) {prf:ref}`prop-orden` |
| `test_compatible_con_pipeline_y_validacion_cruzada` | Funciona dentro de `Pipeline` y `cross_val_score` | integración con scikit-learn |

Las pruebas se ejecutan en cada `push` con GitHub Actions (`.github/workflows/pruebas.yml`).

## Estudio de ablación

```{code-cell} ipython3
:tags: [hide-input]
abl = tabla("ablacion.csv")
a = abl[abl["conjunto"] == "vino"].copy()
nombres = {"completo": "Modelo completo", "sin_agregacion": "Sin agregación (solo umbral 7)", "agregacion_uniforme": "Agregación uniforme",
           "particion_binaria": "Partición binaria", "sin_ordinal": "Sin información ordinal", "sin_calibracion": "Sin calibración"}
a["Variante"] = a["variante"].map(nombres)
cols = {"auc_media": "AUC media", "auc_sd": "DE", "diferencia_vs_completo": "Pérdida frente al completo", "bloques_a_favor_completo": "Bloques a favor del completo",
        "wilcoxon_p_holm": "Wilcoxon (Holm)", "t_p_holm": "t corregida (Holm)", "delong_p_holm": "DeLong (Holm)", "brier_media": "Brier"}
mostrar(a[["Variante"] + list(cols)].rename(columns=cols),
        {"AUC media": "{:.4f}", "DE": "{:.4f}", "Pérdida frente al completo": "{:+.4f}", "Bloques a favor del completo": "{:.0f}",
         "Wilcoxon (Holm)": "{:.4f}", "t corregida (Holm)": "{:.4f}", "DeLong (Holm)": "{:.4f}", "Brier": "{:.4f}"})
```

Cada variante usa los hiperparámetros que eligió la búsqueda del modelo completo en cada pliegue y cambia una sola pieza. Las pruebas son las mismas del capítulo siguiente: Wilcoxon y $t$ corregida sobre los 25 bloques y DeLong sobre las predicciones de la repetición 1, con Holm sobre las cinco variantes.
