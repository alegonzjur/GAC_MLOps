"""
Limpieza y feature engineering del dataset crudo de Kaggle.

Replica la lógica del notebook 01_limpieza_datos.ipynb:
- Corrección de outliers (clip de rangos imposibles)
- Tipado correcto de variables categóricas
- Creación de variables derivadas: risk_flag, gaming_intensity
"""

import pandas as pd

UMBRAL_RIESGO = 50  # Nota por debajo de la cual se considera "en riesgo académico".


def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV crudo tal como se descargó de Kaggle, sin ninguna transformación."""
    return pd.read_csv(path)


def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige valores fuera de rango lógico mediante clipping.
    Se usa clip en vez de eliminar filas, para no perder el resto de información
    de esos registros (que en el análisis exploratorio resultó coherente).
    """
    df = df.copy()
    df['grades'] = df['grades'].clip(upper=100)
    df['addiction_score'] = df['addiction_score'].clip(lower=0)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las variables categóricas a sus tipos correctos (nominal u ordinal)."""
    df = df.copy()

    df['gender'] = df['gender'].astype('category')
    df['gaming_genre'] = df['gaming_genre'].astype('category')

    stress_order = pd.CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=True)
    df['stress_level'] = df['stress_level'].astype(stress_order)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables derivadas necesarias para el modelado posterior."""
    df = df.copy()

    # Target de clasificación: estudiante en riesgo académico.
    df['risk_flag'] = (df['grades'] < UMBRAL_RIESGO).astype(int)

    # Segmentación interpretable de horas de gaming (usada en EDA y Power BI).
    df['gaming_intensity'] = pd.cut(
        df['gaming_hours'],
        bins=[-0.01, 1, 3, 6, 8],
        labels=['Mínimo (0–1h)', 'Moderado (1–3h)', 'Intenso (3–6h)', 'Extremo (6–8h)']
    )

    return df


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de limpieza: encadena las correcciones, tipado
    y feature engineering en el orden correcto.
    """
    df = fix_outliers(df_raw)
    df = fix_dtypes(df)
    df = engineer_features(df)
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Guarda el dataset limpio como checkpoint de transparencia (no usado por el pipeline en sí)."""
    df.to_csv(path, index=False)