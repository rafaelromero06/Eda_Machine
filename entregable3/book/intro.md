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

# Calidad del vino: dashboard predictivo y Bosque Ordinal Focalizado

**Roger Daza, Matteo Del Sole y Rafael Romero** · Machine Learning, Universidad del Norte · Entregable 3, octubre de 2026

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
from myst_nb import glue
from src.libro import tabla, json_tabla, ORIGINAL
v = json_tabla("veredicto.json")
res = tabla("resumen_modelos.csv").set_index("modelo")
glue("auc_bof", f"{v['auc_bof']:.4f}", display=False)
glue("auc_mejor", f"{v['auc_mejor_benchmark']:.4f}", display=False)
glue("mejor", v["mejor_benchmark"], display=False)
glue("sd_bof", f"{res.loc[ORIGINAL, 'auc_sd']:.4f}", display=False)
glue("veredicto", "superior" if v["superior"] else "no superior", display=False)
glue("gana_conj", f"{v['conjuntos_donde_bof_gana']} de 4", display=False)
```

## Resumen ejecutivo

Este informe documenta la tercera y última entrega del proyecto: llevar a producción los mejores modelos del Entregable 2 y proponer un modelo propio validado con un protocolo pre-registrado. El problema es clasificar si un vino Vinho Verde es bueno (nota sensorial de 7 o más, el 19 % de los 5320 vinos sin duplicados) a partir de 11 mediciones fisicoquímicas y del tipo de vino.

El modelo original, el **Bosque Ordinal Focalizado (BOF)**, parte de una limitación de todos los modelos anteriores: aprenden con la etiqueta binaria y desperdician la información de la nota ordinal. El BOF aprende todas las probabilidades acumuladas P(nota ≥ t) con árboles extremadamente aleatorizados de salida múltiple, combina las de los umbrales vecinos al 7 con un núcleo y calibra el puntaje con los casos fuera de bolsa. Está programado desde cero con numpy y scipy, tiene sustento teórico demostrado y 26 pruebas unitarias.

En la validación anidada 5 × 3 repetida 5 veces, el BOF obtuvo un AUC medio de {glue:text}`auc_bof` ± {glue:text}`sd_bof`, frente a {glue:text}`auc_mejor` del mejor de los 9 benchmarks ({glue:text}`mejor`). Según la regla de superioridad fijada antes de los experimentos, el modelo es **{glue:text}`veredicto`**; en los cuatro conjuntos adicionales supera al mejor benchmark en {glue:text}`gana_conj`. PENDIENTE: una frase con la conclusión principal.

## Problema y motivación

La certificación de la calidad del vino depende de paneles de catadores: es lenta, costosa y subjetiva. Las pruebas fisicoquímicas, en cambio, son rápidas y baratas, y {cite:t}`cortez2009` mostraron que permiten predecir la nota sensorial. En 2023 la producción mundial de vino fue la más baja desde 1961 {cite:p}`oiv2024`; con menos producción, identificar pronto los lotes de calidad tiene valor para los productores.

El Entregable 2 comparó 136 combinaciones de 7 modelos, 4 técnicas de balanceo y 4 optimizadores con validación anidada, y encontró que el modelo importa más que el ajuste: Random Forest y XGBoost lideran y el techo de AUC ronda 0.86. Todos esos modelos ven solo la etiqueta binaria.

## Contribuciones

1. **Correcciones y auditoría.** Se atendieron todas las observaciones del Entregable 2 (capítulo 1) y se auditó con código cada vía de fuga de información (capítulo 2).
2. **Selección estadística y artefactos.** La mejor combinación de cada modelo se eligió por el AUC medio externo, se reentrenó con todos los datos y se serializó con metadatos y prueba de ida y vuelta (capítulo 3).
3. **Un mecanismo nuevo** para clasificar un evento definido por el corte de una respuesta ordinal: un criterio de partición que es una regla de puntuación propia centrada en el umbral, una combinación focalizada de umbrales vecinos y una calibración fuera de bolsa, con demostraciones de sus propiedades (capítulos 4 y 5).
4. **Una evaluación pre-registrada y amplia:** 25 bloques pareados, 9 benchmarks (incluidos LightGBM y CatBoost), cuatro pruebas de hipótesis con corrección de Holm, tamaños de efecto, ablación, cuatro conjuntos adicionales y ocho estudios de robustez (capítulos 6 y 7).
5. **Un dashboard desplegado** con predicción en vivo, incertidumbre conforme y explicaciones SHAP, también para el modelo original (capítulo 8).

## Objetivos

- Verificar con rigor estadístico si aprovechar la nota ordinal mejora la detección de vinos buenos.
- Implementar ese mecanismo como un modelo propio, compatible con scikit-learn y defendible línea por línea.
- Poner los mejores modelos a disposición de un usuario no experto.

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 1 | Trazabilidad de las correcciones del Entregable 2 |
| 2 | Datos: ETL, EDA, declaración de los datos y auditoría de fuga |
| 3 | Modelos benchmark, selección de las mejores combinaciones y artefactos |
| 4 | Revisión de literatura y declaración de novedad |
| 5 | El modelo original: formulación, teoría, complejidad, implementación, pruebas y ablación |
| 6 | Protocolo pre-registrado y pruebas de hipótesis |
| 7 | Conjuntos adicionales, robustez, calibración, incertidumbre, errores e interpretabilidad |
| 8 | Dashboard y despliegue |
| 9 | Conclusiones, limitaciones y declaración de uso de herramientas |
| 10 | Reproducibilidad, tabla maestra y bibliografía |

**Dashboard:** PENDIENTE (URL pública de Render una vez desplegado). **Código:** https://github.com/rafaelromero06/Eda_Machine (carpeta `entregable3/`).
