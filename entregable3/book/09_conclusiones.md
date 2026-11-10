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

La química permite anticipar si un vino Vinho Verde obtendrá una nota de 7 o más con un AUC cercano a 0.86: de cada 100 parejas formadas por un vino bueno y uno que no lo es, los mejores modelos ordenan bien unas 86. Ninguna variable basta por sí sola (la correlación más alta con la nota es de 0.47, la del alcohol) y la dificultad se concentra en la frontera entre las notas 6 y 7, que reúne al 60 % de los vinos.

Entre los 9 benchmarks, CatBoost y Random Forest son los mejores y quedan a menos de una desviación estándar entre sí. La elección del modelo importa mucho más que la del balanceo o la del optimizador, como ya mostraba el Entregable 2.

El Bosque Ordinal Focalizado obtiene el AUC medio más alto de los 10 modelos (0.8649 frente a 0.8631 de CatBoost), supera con las tres pruebas pareadas a 7 de los 9 benchmarks y es el mejor calibrado junto con Random Forest. Frente a CatBoost y Random Forest la diferencia no se distingue del azar de las particiones. Sus ventajas más claras no están en el AUC medio: es el modelo fuerte que mejor se transfiere de tintos a blancos y viceversa, el que menos pierde cuando las notas tienen ruido y el mejor con pocos datos. Sus costos son el tiempo de ajuste y la memoria.

La ablación indica que ninguna pieza del mecanismo cambia mucho el resultado por sí sola: quitar la combinación de umbrales produce la mayor caída (0.0037 de AUC) y la única caída confirmada por las pruebas es la de dar el mismo peso a todos los umbrales (0.0018). La calibración no cambia el AUC, pero reduce el puntaje de Brier de 0.139 a 0.108.

## Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| R1. Novedad en la aplicación | Se cumple, con la evidencia disponible | Capítulo 4: búsqueda en 4 bases, 8 cadenas, citas desde 6 trabajos y repositorios; ningún trabajo usa el mecanismo para la calidad del vino ni para otro evento binario definido por un corte ordinal |
| R2. Implementación algorítmica propia | Se cumple | `src/original_model.py`: crecimiento de los árboles, criterio de partición, agregación y calibración programados con numpy y scipy; 26 pruebas unitarias |
| R3. Superioridad estadística | El modelo es **{glue:text}`veredicto`** según la regla pre-registrada | Capítulo 6 |

El resultado de R3 es negativo y se reporta como tal. La regla exigía cuatro condiciones a la vez y el BOF cumple dos: tiene el mejor AUC medio y un tamaño de efecto pareado no despreciable, pero solo dos de las cuatro pruebas rechazan la igualdad frente a CatBoost y no gana en ninguno de los cuatro conjuntos adicionales. La afirmación de novedad (R1) descansa en una búsqueda que no cubrió bases de pago; no prueba que el mecanismo no exista y debe repetirse antes de la entrega final.

Hay tres causas probables. La primera es un techo impuesto por los datos: los tres mejores modelos, muy distintos entre sí, llegan al mismo AUC, y el error se concentra en la frontera entre 6 y 7, donde la química casi no distingue a los vinos y la nota, mediana de pocos catadores, es ruidosa. La información ordinal ayuda a ordenar los extremos de la escala, pero no esa frontera. La segunda es de potencia: con 25 bloques que comparten vinos, la t corregida necesita diferencias mucho mayores que 0.002 de AUC para rechazar. La tercera es de diseño: la ablación muestra que el mecanismo aporta poco sobre un bosque de árboles extremadamente aleatorizados, y la búsqueda de hiperparámetros no encontró la mejor configuración (el estudio de sensibilidad obtiene un AUC mayor con menos variables candidatas por corte). En los conjuntos adicionales, más pequeños o con menos niveles en la respuesta, el BOF queda entre los puestos 2 y 5: compite, pero no hay evidencia de que el mecanismo se generalice como ventaja.

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
