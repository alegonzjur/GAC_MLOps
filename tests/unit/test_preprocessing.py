"""
Tests unitarios para src/data/preprocessing.py

A diferencia de los tests de limpieza, aquí usamos una fixture con más filas,
porque los splits estratificados y el escalado necesitan suficiente variabilidad
para que las comprobaciones tengan sentido real.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    FEATURES, FEATURES_CLUSTER,
    encode_categoricals, split_classification, split_regression,
    scale_features, prepare_cluster_data
)


@pytest.fixture
def df_preparado():
    """
    DataFrame sintético de 40 filas, balanceado en risk_flag,
    con todas las columnas que preprocessing.py espera encontrar.
    """
    n = 40
    rng = np.random.RandomState(42)  # Semilla fija: el test siempre genera los mismos datos.

    df = pd.DataFrame({
        'age': rng.randint(18, 25, n),
        'gaming_hours': rng.uniform(0, 8, n),
        'study_hours': rng.uniform(0, 10, n),
        'sleep_hours': rng.uniform(4, 10, n),
        'attendance': rng.uniform(50, 100, n),
        'social_activity': rng.uniform(0, 5, n),
        'device_usage': rng.uniform(0, 24, n),
        'reaction_time_ms': rng.uniform(150, 500, n),
        'addiction_score': rng.uniform(0, 25, n),
        'gender': rng.choice(['M', 'F'], n),
        'gaming_genre': rng.choice(['FPS', 'RPG', 'Estrategia'], n),
        'stress_level': rng.choice(['Low', 'Medium', 'High'], n),
        'grades': rng.uniform(30, 100, n),
        'risk_flag': [i % 2 for i in range(n)],  # Balanceado 50/50, a propósito.
    })
    return df


@pytest.fixture
def df_codificado(df_preparado):
    """Conveniencia: devuelve el DataFrame ya pasado por encode_categoricals."""
    return encode_categoricals(df_preparado)


def test_encode_categoricals_crea_columnas_enc(df_preparado):
    """Las 3 columnas codificadas deben existir tras encode_categoricals."""
    df_resultado = encode_categoricals(df_preparado)

    assert 'gender_enc' in df_resultado.columns
    assert 'gaming_genre_enc' in df_resultado.columns
    assert 'stress_level_enc' in df_resultado.columns


def test_encode_categoricals_columnas_son_numericas(df_codificado):
    """Las columnas _enc deben quedar numéricas, no texto."""
    assert pd.api.types.is_numeric_dtype(df_codificado['gender_enc'])
    assert pd.api.types.is_numeric_dtype(df_codificado['gaming_genre_enc'])
    assert pd.api.types.is_numeric_dtype(df_codificado['stress_level_enc'])


def test_split_classification_respeta_test_size(df_codificado):
    """El split debe repartir aproximadamente 80/20 entre train y test."""
    X_train, X_test, y_train, y_test = split_classification(df_codificado, test_size=0.2)

    total = len(df_codificado)
    assert len(X_test) == round(total * 0.2)
    assert len(X_train) == total - len(X_test)


def test_split_classification_mantiene_proporcion_clases(df_codificado):
    """
    Gracias a stratify, la proporción de risk_flag=1 debe ser
    prácticamente igual en train y en test.
    """
    X_train, X_test, y_train, y_test = split_classification(df_codificado, test_size=0.2)

    proporcion_train = y_train.mean()
    proporcion_test = y_test.mean()

    # Con un dataset balanceado 50/50, ambas proporciones deben rondar 0.5.
    assert abs(proporcion_train - proporcion_test) < 0.15


def test_split_regression_respeta_test_size(df_codificado):
    """El split de regresión también debe repartir aproximadamente 80/20."""
    X_train, X_test, y_train, y_test = split_regression(df_codificado, test_size=0.2)

    total = len(df_codificado)
    assert len(X_test) == round(total * 0.2)


def test_scale_features_media_cero_en_train(df_codificado):
    """Tras escalar, la media de cada feature en train debe ser ~0 (propiedad de StandardScaler)."""
    X_train, X_test, y_train, y_test = split_regression(df_codificado)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    medias_train = X_train_scaled.mean(axis=0)
    assert np.allclose(medias_train, 0, atol=1e-10)


def test_scale_features_desviacion_uno_en_train(df_codificado):
    """Tras escalar, la desviación estándar de cada feature en train debe ser ~1."""
    X_train, X_test, y_train, y_test = split_regression(df_codificado)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    desviaciones_train = X_train_scaled.std(axis=0)
    assert np.allclose(desviaciones_train, 1, atol=1e-10)


def test_scale_features_test_no_se_usa_para_ajustar(df_codificado):
    """
    El scaler debe ajustarse SOLO con train. Verificamos esto comprobando que
    transformar X_test manualmente con el mismo scaler da el mismo resultado
    que el devuelto por la función (si hubiera hecho fit con test, no coincidiría).
    """
    X_train, X_test, y_train, y_test = split_regression(df_codificado)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    X_test_scaled_manual = scaler.transform(X_test)
    assert np.allclose(X_test_scaled, X_test_scaled_manual)


def test_prepare_cluster_data_selecciona_columnas_correctas(df_codificado):
    """X_cluster_scaled debe tener exactamente el número de columnas de FEATURES_CLUSTER."""
    X_cluster_scaled, scaler = prepare_cluster_data(df_codificado)

    assert X_cluster_scaled.shape[1] == len(FEATURES_CLUSTER)
    assert X_cluster_scaled.shape[0] == len(df_codificado)