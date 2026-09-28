
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import config


def cargar_datos(ruta_data: Path = config.RUTA_DATA) -> pd.DataFrame:
    """Carga winequality-red.csv y winequality-white.csv, los une y agrega
    la columna ``tipo_vino``. No quita duplicados todavía (eso se hace
    explícitamente en :func:`preparar_dataset` para que quede trazable
    cuántas filas se pierden)."""
    red = pd.read_csv(ruta_data / "winequality-red.csv", sep=";")
    white = pd.read_csv(ruta_data / "winequality-white.csv", sep=";")
    red["tipo_vino"] = "red"
    white["tipo_vino"] = "white"
    df = pd.concat([red, white], ignore_index=True)
    return df


def preparar_dataset(ruta_data: Path = config.RUTA_DATA, verbose: bool = True) -> pd.DataFrame:
    """Carga + quita duplicados. Devuelve el dataframe listo para separar
    en X / y_reg / y_clf."""
    df = cargar_datos(ruta_data)
    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_despues = len(df)
    if verbose:
        print(f"Filas antes de quitar duplicados: {n_antes}")
        print(f"Filas después:                    {n_despues}  "
              f"(-{n_antes - n_despues}, {(n_antes - n_despues) / n_antes:.1%})")
    return df


def separar_X_y(df: pd.DataFrame):
    """Devuelve (X, y_reg, y_clf) a partir del dataframe ya sin duplicados."""
    X = df[config.COLUMNAS_NUMERICAS + config.COLUMNAS_CATEGORICAS].copy()
    y_reg = df["quality"].astype(float).copy()
    y_clf = (df["quality"] >= config.UMBRAL_CLASIFICACION).astype(int).copy()
    return X, y_reg, y_clf


def construir_preprocesador() -> ColumnTransformer:
    """ColumnTransformer: StandardScaler para numéricas, OneHotEncoder para
    tipo_vino. Se construye "fresco" cada vez que se llama (nunca se
    comparte una instancia ya ajustada entre pipelines) para que cada
    Pipeline downstream la ajuste únicamente con los datos que le
    correspondan en su propio fold."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), config.COLUMNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.COLUMNAS_CATEGORICAS),
    ])


if __name__ == "__main__":
    df = preparar_dataset()
    X, y_reg, y_clf = separar_X_y(df)
    print(f"\nX: {X.shape}")
    print(f"Balance clasificación (quality>={config.UMBRAL_CLASIFICACION}): "
          f"{y_clf.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Distribución quality (regresión):\n{y_reg.value_counts().sort_index()}")
