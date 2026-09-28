
import time
import warnings

warnings.filterwarnings("ignore")

from src import config
from src.balance_factory import NOMBRES_BALANCEO
from src.data_prep import preparar_dataset, separar_X_y
from src.model_factory import ESPACIOS_CLASIFICACION, ESPACIOS_REGRESION
from src.nested_cv import ejecutar_en_paralelo
from src.optimizers import OPTIMIZADORES


def combinaciones_clasificacion():
    return [(m, b, o) for m in ESPACIOS_CLASIFICACION for b in NOMBRES_BALANCEO for o in OPTIMIZADORES]


def combinaciones_regresion():
    # "no_aplica" en la columna balanceo, consistente con la Sección 2:
    # el balanceo de clases se omite por completo en regresión.
    return [(m, "no_aplica", o) for m in ESPACIOS_REGRESION for o in OPTIMIZADORES]


def main():
    print(f"=== INICIO {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"Presupuesto de evaluaciones: {config.PRESUPUESTO_EVALUACIONES} | "
          f"K_OUTER={config.K_OUTER} | K_INNER={config.K_INNER} | "
          f"procesos en paralelo={config.N_WORKERS}", flush=True)
    print(f"Resultados en: {config.RUTA_LOGS}\n", flush=True)

    df = preparar_dataset(verbose=True)
    X, y_reg, y_clf = separar_X_y(df)
    print()

    grupos = [
        {"tarea": "clasificacion", "combinaciones": combinaciones_clasificacion(),
         "X": X, "y": y_clf, "ruta_log": config.RUTA_LOG_CLASIFICACION},
        {"tarea": "regresion", "combinaciones": combinaciones_regresion(),
         "X": X, "y": y_reg, "ruta_log": config.RUTA_LOG_REGRESION},
    ]
    try:
        ejecutar_en_paralelo(grupos)
    except KeyboardInterrupt:
        print("\n[interrumpido] Los folds terminados ya quedaron guardados en los CSV. "
              "Vuelve a correr este mismo comando para continuar donde quedó.", flush=True)
        return

    print(f"\n=== FIN {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":  # obligatorio en Windows para usar varios procesos
    main()
