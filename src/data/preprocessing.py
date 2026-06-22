"""
Carga y preprocesamiento del dataset GamingAcademicPerformance.

Replica la lógica de las celdas 2-11 del notebook original (03_modelo_ml.ipynb),
convertida en funciones reutilizables para el pipeline de entrenamiento.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Columnas usadas como features para clasificación y regresión.
FEATURES = [
    'age', 'gaming_hours', 'study_hours', 'sleep_hours',
    'attendance', 'social_activity', 'device_usage',
    'reaction_time_ms', 'addiction_score',
    'gender_enc', 'gaming_genre_enc', 'stress_level_enc'
]

# Columnas usadas específicamente para clustering (sin target).
FEATURES_CLUSTER = [
    'gaming_hours', 'study_hours', 'sleep_hours',
    'attendance', 'addiction_score', 'reaction_time_ms',
    'stress_level_enc', 'social_activity'
]

TARGET_CLASS = 'risk_flag'
TARGET_REG = 'grades'


def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el CSV y restaura los tipos categóricos ordinales."""
    df = pd.read_csv(path)

    stress_order = pd.CategoricalDtype(
        categories=['Low', 'Medium', 'High'], ordered=True
    )
    df['stress_level'] = df['stress_level'].astype(stress_order)

    intensity_order = pd.CategoricalDtype(
        categories=['Mínimo (0–1h)', 'Moderado (1–3h)', 'Intenso (3–6h)', 'Extremo (6–8h)'],
        ordered=True
    )
    df['gaming_intensity'] = df['gaming_intensity'].astype(intensity_order)

    return df


def encode_categoricals(df: pd.DataFrame, return_encoders: bool = False):
    df = df.copy()

    le_gender = LabelEncoder()
    le_genre = LabelEncoder()
    le_stress = LabelEncoder()

    df['gender_enc'] = le_gender.fit_transform(df['gender'].astype(str))
    df['gaming_genre_enc'] = le_genre.fit_transform(df['gaming_genre'].astype(str))
    df['stress_level_enc'] = le_stress.fit_transform(df['stress_level'].astype(str))

    if return_encoders:
        encoders = {
            'gender': le_gender,
            'gaming_genre': le_genre,
            'stress_level': le_stress,
        }
        return df, encoders

    return df


def split_classification(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train/test split para el problema de clasificación (estratificado)."""
    X = df[FEATURES]
    y = df[TARGET_CLASS]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_regression(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train/test split para el problema de regresión."""
    X = df[FEATURES]
    y = df[TARGET_REG]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Escala features con StandardScaler.
    El scaler se ajusta SOLO con train para evitar data leakage,
    y se aplica (transform) igual sobre test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_cluster_data(df: pd.DataFrame):
    """Selecciona y escala las features específicas para clustering."""
    X_cluster = df[FEATURES_CLUSTER].copy()
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    return X_cluster_scaled, scaler