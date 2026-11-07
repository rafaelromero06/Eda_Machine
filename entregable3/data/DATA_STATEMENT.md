# Declaración de los datos

## Conjunto principal: Wine Quality

| Aspecto | Detalle |
|---|---|
| Procedencia | Comisión de Viticultura de la Región de los Vinhos Verdes (Portugal), publicado en el UCI Machine Learning Repository (conjunto 186). |
| Referencia | Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553. |
| Licencia | Creative Commons Attribution 4.0 (CC BY 4.0). |
| Periodo | Muestras analizadas entre mayo de 2004 y febrero de 2007. |
| Contenido | 6497 vinos (1599 tintos y 4898 blancos), 11 variables fisicoquímicas medidas en laboratorio y la nota sensorial (mediana de al menos tres catadores, escala de 0 a 10). |
| Procesamiento | Se unen tintos y blancos con la variable `tipo_vino` y se eliminan 1177 filas duplicadas exactas, que quedan 5320 vinos. El evento de interés es `quality >= 7` (18.97 % de los vinos). No hay valores faltantes. |
| Archivos | `winequality-red.csv` y `winequality-white.csv` (separador `;`), sin modificar. |

### Sesgos y limitaciones conocidos

- **Una sola región y época.** Solo hay vinos Vinho Verde de 2004 a 2007; no hay garantía de que las relaciones valgan para otras regiones, uvas o añadas.
- **Desbalance de tipos.** Hay casi tres blancos por cada tinto; el modelo ve menos ejemplos de tintos buenos.
- **Desbalance de clases.** Solo 1 de cada 5 vinos es bueno y las notas 3 y 9 son muy raras.
- **Etiqueta subjetiva.** La nota es la mediana de catadores; dos vinos con la misma química pueden tener notas distintas.
- **Variables ausentes.** No hay uva, precio, marca ni añada; por privacidad del productor tampoco hay identificadores de lote. El análisis de residuos del Entregable 2 sugiere que vinos consecutivos de los archivos se parecen más de lo que explica la química (posibles lotes), lo que puede hacer algo optimista la validación aleatoria.

### Usos inadecuados

- Reemplazar la evaluación sensorial o la certificación oficial.
- Fijar precios, rechazar lotes o tomar decisiones comerciales sin una validación local.
- Hacer recomendaciones de consumo de alcohol o de salud.

## Conjuntos adicionales (solo para la evaluación del modelo original)

Todos provienen del UCI Machine Learning Repository con licencia CC BY 4.0 y se usan sin modificar, salvo quitar filas duplicadas exactas.

| Conjunto | DOI | Evento | Nota ética |
|---|---|---|---|
| Abalone (Nash et al., 1994) | 10.24432/C55C7W | anillos ≥ 11 | Datos biológicos, sin personas. |
| Cardiotocography (Ayres-de Campos et al., 2000) | 10.24432/C51S4N | NSP ≥ 2 (sospechoso o patológico) | Registros clínicos anonimizados; el modelo no está validado para uso clínico. |
| Student Performance, portugués (Cortez y Silva, 2008) | 10.24432/C5TG7T | G3 ≥ 10 (aprobado) | Contiene sexo, edad y datos familiares de estudiantes: no debe usarse para decidir sobre personas. Se excluyen G1 y G2. |
| Heart Disease, Cleveland (Detrano et al., 1989) | 10.24432/C52P4X | num ≥ 1 (enfermedad) | Datos clínicos con sexo y edad; 6 valores faltantes imputados dentro de cada pliegue. Uso solo comparativo. |

Los cuatro se eligieron antes de evaluar cualquier modelo en ellos, con el criterio fijado en `experiments/protocolo.md`: clasificación binaria obtenida al cortar una respuesta ordinal que viene en los propios datos. La procedencia detallada está en `data/adicionales/procedencia.json`.
