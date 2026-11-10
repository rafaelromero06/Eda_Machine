import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.data import ADICIONALES

TABLAS = RAIZ / "experiments" / "resultados" / "tablas"
META = RAIZ / "artifacts" / "bosque_ordinal_focalizado" / "metadata.json"


def main():
    v = json.loads((TABLAS / "veredicto.json").read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    res = pd.read_csv(TABLAS / "resumen_modelos.csv").set_index("modelo")
    cal = pd.read_csv(TABLAS / "calibracion.csv").set_index("modelo")
    adic = pd.read_csv(TABLAS / "adicionales_pruebas.csv")
    rob = pd.read_csv(TABLAS / "robustez_resumen.csv").set_index("Modelo")
    bof = res.loc["Bosque Ordinal Focalizado"]
    mejor = v["mejor_benchmark"]
    hp = meta["hiperparametros_finales"]
    red = meta.get("reduccion_arboles", {})
    filas_adic = "\n".join(f"| {ADICIONALES[r.conjunto]['titulo']} | {r.auc_bof:.3f} | {r.mejor_benchmark} ({r.auc_mejor:.3f}) | "
                           f"{int(r.puesto_bof)} de 10 |" for r in adic.itertuples())
    veredicto = ("**superior** según la regla pre-registrada" if v["superior"] else
                 "**no superior** según la regla pre-registrada (el detalle de las pruebas está en el capítulo 6 del informe)")
    ref_bof = rob.loc["Bosque Ordinal Focalizado", "AUC de referencia"]
    bloques_bof = rob.loc["Bosque Ordinal Focalizado"].get("AUC con bloques del archivo", float("nan"))
    if pd.isna(bloques_bof):
        aviso_bloques = "- No se pudo estimar el efecto de los lotes: el conjunto no identifica el lote de cada vino."
    elif ref_bof - bloques_bof > 0.005:
        aviso_bloques = (f"- Con pliegues formados por bloques de 10 vinos consecutivos del archivo (posibles lotes), el AUC baja de "
                         f"{ref_bof:.3f} a {bloques_bof:.3f}: los vinos de un mismo lote comparten información, así que con lotes "
                         f"nuevos el desempeño puede ser algo menor al reportado.")
    else:
        aviso_bloques = (f"- Con pliegues formados por bloques de 10 vinos consecutivos del archivo (posibles lotes), el AUC se mantiene "
                         f"({bloques_bof:.3f} frente a {ref_bof:.3f}); no hay indicios de que los lotes inflen el desempeño reportado, "
                         f"aunque el conjunto no identifica el lote de cada vino.")
    texto = f"""# Ficha del modelo: Bosque Ordinal Focalizado

## Detalles del modelo

- **Desarrolladores:** Roger Daza, Matteo Del Sole y Rafael Romero (Universidad del Norte), curso de Machine Learning, 2026.
- **Versión:** 1.0 ({meta['fecha']}). Artefacto: `artifacts/bosque_ordinal_focalizado/modelo.joblib` (SHA-256 `{meta['sha256'][:16]}…`).
- **Tipo:** clasificador binario basado en un bosque de árboles extremadamente aleatorizados de salida múltiple, programado desde cero con numpy y scipy (`src/original_model.py`).
- **Algoritmo:** aprende las probabilidades acumuladas P(nota ≥ t) con una impureza ponderada por un núcleo centrado en el umbral 7, puntúa combinando los umbrales vecinos con un segundo núcleo y calibra el puntaje con los casos fuera de bolsa.
- **Hiperparámetros finales:** {", ".join(f"`{k}` = {val:.4g}" if isinstance(val, float) else f"`{k}` = {val}" for k, val in hp.items())}. Se despliegan {red.get('arboles_desplegados', hp.get('n_estimators'))} árboles (el menor número con AUC fuera de bolsa a menos de 0.001 del bosque completo).
- **Licencia:** MIT. **Cita:** ver `CITATION.cff`.
- **Contacto:** repositorio https://github.com/rafaelromero06/Eda_Machine.

## Uso previsto

- **Uso principal:** estimar la probabilidad de que un vino Vinho Verde reciba una nota sensorial de 7 o más a partir de 11 mediciones fisicoquímicas de laboratorio y del tipo de vino, como apoyo para priorizar lotes antes de la cata.
- **Usuarios previstos:** enólogos, laboratorios de control de calidad y estudiantes que quieran entender la relación entre química y calidad percibida.
- **Usos no recomendados:** reemplazar la evaluación sensorial o la certificación oficial; fijar precios o rechazar lotes sin validación local; aplicarlo a vinos de otras regiones, uvas o épocas sin reentrenar; cualquier recomendación de consumo de alcohol o de salud.

## Factores

- **Tipo de vino:** los blancos son casi tres veces más que los tintos y tienen más vinos buenos; el desempeño se reporta por tipo.
- **Región y época:** solo vinos Vinho Verde de 2004 a 2007.
- **Etiqueta:** la nota es la mediana de al menos tres catadores; es subjetiva y discreta.

## Métricas

- **Principal:** AUC-ROC del evento en el bucle externo de la validación anidada 5 × 3 repetida 5 veces (25 particiones).
- **Secundarias:** PR-AUC, puntaje de Brier, log-loss, ECE, F1 con umbral 0.5 y cobertura de los conjuntos conformes.
- **Variación:** media ± desviación estándar en las 25 particiones e intervalo bootstrap del 95 % sobre las predicciones fuera de muestra.

## Datos de evaluación y de entrenamiento

- **Entrenamiento del artefacto:** los 5320 vinos sin duplicados del conjunto Wine Quality (UCI, CC BY 4.0), con la nota completa como respuesta.
- **Evaluación:** las mismas 25 particiones para todos los modelos; además cuatro conjuntos públicos adicionales (Abalone, Cardiotocografía, Student Performance y Heart Disease) elegidos antes de evaluar con el criterio del protocolo.
- **Preprocesamiento:** escalado estándar y one-hot del tipo de vino dentro de cada pliegue.

## Resultados cuantitativos

| Métrica | Bosque Ordinal Focalizado | Mejor benchmark ({mejor}) |
|---|---|---|
| AUC (media ± DE, 25 particiones) | {bof.auc_media:.4f} ± {bof.auc_sd:.4f} | {res.loc[mejor, 'auc_media']:.4f} ± {res.loc[mejor, 'auc_sd']:.4f} |
| AUC fuera de muestra, repetición 1 (IC 95 %) | {bof.auc_oof_rep1:.4f} ({bof.auc_ic_inf:.4f} a {bof.auc_ic_sup:.4f}) | {res.loc[mejor, 'auc_oof_rep1']:.4f} ({res.loc[mejor, 'auc_ic_inf']:.4f} a {res.loc[mejor, 'auc_ic_sup']:.4f}) |
| PR-AUC | {bof.pr_auc_media:.4f} | {res.loc[mejor, 'pr_auc_media']:.4f} |
| Brier | {bof.brier_media:.4f} | {res.loc[mejor, 'brier_media']:.4f} |
| ECE | {cal.loc['Bosque Ordinal Focalizado', 'ece']:.4f} | {cal.loc[mejor, 'ece'] if mejor in cal.index else float('nan'):.4f} |
| Cobertura del conjunto conforme al 90 % | {rob.loc['Bosque Ordinal Focalizado', 'Cobertura conforme 90 %']:.3f} | {rob.loc[mejor, 'Cobertura conforme 90 %']:.3f} |

**Veredicto de superioridad (R3):** el modelo es {veredicto}. Diferencia de AUC media frente al mejor benchmark: {v['auc_bof'] - v['auc_mejor_benchmark']:+.4f}; δ de Cliff = {v['delta_cliff']:.2f}.

**Conjuntos adicionales (AUC media de 5 pliegues):**

| Conjunto | BOF | Mejor benchmark | Puesto del BOF |
|---|---|---|---|
{filas_adic}

## Consideraciones éticas

- La nota sensorial refleja las preferencias de un panel concreto; el modelo reproduce esas preferencias, no una calidad objetiva.
- Un error del modelo puede descartar un lote bueno o promover uno mediocre: debe usarse como apoyo y no como decisión automática.
- Los conjuntos adicionales de salud y educación se usaron solo para comparar algoritmos; el modelo no está validado para decisiones sobre personas.

## Advertencias y recomendaciones

{aviso_bloques}
- La probabilidad calibrada es confiable dentro del rango de los datos de entrenamiento; fuera de los rangos válidos del esquema el formulario del dashboard rechaza la entrada.
- Para usar el modelo en otra región o época, se debe reentrenar y volver a validar con el mismo protocolo.
"""
    (RAIZ / "MODEL_CARD.md").write_text(texto, encoding="utf-8")
    print("MODEL_CARD.md generado")


if __name__ == "__main__":
    main()
