
from imblearn.pipeline import Pipeline as ImbPipeline

from .data_prep import construir_preprocesador
from .model_factory import nuevo_estimador
from .balance_factory import aplicar_balanceo


def construir_pipeline(tarea: str, nombre_modelo: str, nombre_balanceo: str = "sin_balanceo"):
    """Devuelve (pipeline, aplica). `aplica=False` si la combinación
    (modelo, balanceo) no tiene sentido (ver balance_factory) y por lo
    tanto no debe incluirse en la grilla de experimentos."""
    estimador = nuevo_estimador(tarea, nombre_modelo)

    pasos_muestreo = []
    aplica = True
    if tarea == "clasificacion":
        estimador, pasos_muestreo, aplica = aplicar_balanceo(nombre_balanceo, nombre_modelo, estimador)

    pasos = [("prep", construir_preprocesador())] + pasos_muestreo + [("modelo", estimador)]
    return ImbPipeline(pasos), aplica
