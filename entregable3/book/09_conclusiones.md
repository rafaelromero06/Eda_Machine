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

# Conclusiones y limitaciones

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib
RAIZ = pathlib.Path.cwd().parent
sys.path.insert(0, str(RAIZ))
from myst_nb import glue
from src.libro import tabla, json_tabla
v = json_tabla("veredicto.json")
glue("veredicto", "superior" if v["superior"] else "no superior", display=False)
```

## Hallazgos

PENDIENTE.

## Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| R1. Novedad en la aplicación | PENDIENTE | Capítulo 4: búsqueda en 4 bases, 8 cadenas, citas desde 6 trabajos y repositorios; ningún trabajo usa el mecanismo para la calidad del vino ni para otro evento binario definido por un corte ordinal |
| R2. Implementación algorítmica propia | Se cumple | `src/original_model.py`: crecimiento de los árboles, criterio de partición, agregación y calibración programados con numpy y scipy; 26 pruebas unitarias |
| R3. Superioridad estadística | El modelo es **{glue:text}`veredicto`** según la regla pre-registrada | Capítulo 6 |

PENDIENTE: discusión honesta del resultado de R3 y su causa probable.

## Limitaciones

- **Datos.** Un solo productor regional (Vinho Verde), vinos de 2004 a 2007 y una etiqueta subjetiva: la mediana de al menos tres catadores. El modelo aprende las preferencias de ese panel, no una calidad objetiva.
- **Lotes no identificados.** No hay una variable de lote o de productor. La validación por bloques del archivo ({ref}`sec-auditoria`) sugiere que los vinos vecinos comparten información, así que el AUC de todos los modelos puede ser algo optimista para lotes nuevos.
- **Dependencia entre bloques.** Los 25 bloques comparten vinos. La *t* corregida lo tiene en cuenta, pero Wilcoxon, Friedman y Nemenyi suponen bloques independientes y se interpretan como aproximaciones.
- **Pocos conjuntos para generalizar.** Con cinco conjuntos, la comparación entre conjuntos tiene poca potencia; Wilcoxon no puede bajar de p = 0.0625.
- **Alcance del mecanismo.** El BOF solo tiene sentido cuando el evento es el corte de una respuesta ordinal o numérica; con una etiqueta binaria pura se reduce a un Extra-Trees.
- **Costo.** La implementación está en numpy puro y es más lenta de entrenar que Random Forest de scikit-learn, que está escrito en C.
- **Medición del cómputo.** Los tiempos se midieron con 11 procesos en paralelo en un equipo de escritorio, así que tienen ruido por la competencia entre procesos.

## Riesgos y usos inadecuados

- Usar la probabilidad como certificación de calidad: el modelo reproduce las preferencias de un panel de catadores de una región y una época.
- Aplicarlo a vinos de otras regiones, uvas o años sin reentrenar: el estudio de cambio de distribución muestra cuánto se degrada entre tintos y blancos.
- Decisiones automáticas sin revisión humana: un falso negativo descarta un lote bueno y un falso positivo promueve uno mediocre.
- Los conjuntos adicionales de salud y educación se usaron solo para comparar algoritmos; el modelo no está validado para decisiones sobre personas.

## Trabajo futuro

- Validar con datos que identifiquen el lote o el productor, usando pliegues agrupados.
- Probar el mecanismo con *gradient boosting*: una pérdida de Brier ponderada por el mismo núcleo sobre las probabilidades acumuladas, en lugar de árboles aleatorizados.
- Elegir los anchos de los núcleos con los casos fuera de bolsa durante el ajuste, sin búsqueda externa.
- Aprovechar la distribución predicha de la nota para decisiones con costos (por ejemplo, separar vinos de 8 o más para una línea premium) y para conjuntos conformes ordinales.
- Reescribir el crecimiento de los árboles con numba o Cython para acercar su costo al de Random Forest.

## Declaración de uso de herramientas de inteligencia artificial

En este entregable se usó **Claude (Anthropic)**, un asistente de inteligencia artificial, a través de Claude Code, para:

1. proponer, programar y probar el código del repositorio (el modelo original, los experimentos, las pruebas estadísticas, el dashboard y los scripts de apoyo);
2. ejecutar los experimentos, las pruebas unitarias y la revisión bibliográfica por API;
3. redactar los borradores de este libro, de la guía del código, del protocolo y de la documentación.

Las decisiones de diseño del modelo original se tomaron a partir de experimentos piloto documentados (anexo A del protocolo y `experiments/bitacora_diseno.csv`), y todos los números del libro se calculan con el código del repositorio al compilarlo.

Al firmar, cada integrante declara que revisó el código y el informe, que entiende el funcionamiento del modelo original y de las pruebas, que puede explicarlos y modificarlos en la sustentación, y que asume la responsabilidad por su contenido.

| Integrante | Firma | Fecha |
|---|---|---|
| Roger Daza | | |
| Matteo Del Sole | | |
| Rafael Romero | | |
