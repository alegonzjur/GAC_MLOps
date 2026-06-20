"""
Tests unitarios para src/data/cleaning.py

Cada test comprueba una única responsabilidad de limpieza,
usando DataFrames pequeños y controlados (no el dataset real).
"""

import pandas as pd
import pytest

from src.data.cleaning import fix_outliers, fix_dtypes, engineer_features, clean_data


@pytest.fixture
def df_con_anomalias():
    """DataFrame mínimo con los casos problemáticos que sabemos que existen en los datos reales."""
    return pd.DataFrame({
        'student_id': [1, 2, 3],
        'grades': [118.0, 45.0, 75.0],       # 118 está fuera de rango (> 100).
        'addiction_score': [-5.0, 10.0, 20.0],  # -5 está fuera de rango (< 0).
        'gaming_hours': [0.5, 4.0, 7.0],
        'gender': ['M', 'F', 'M'],
        'gaming_genre': ['FPS', 'RPG', 'FPS'],
        'stress_level': ['Low', 'Medium', 'High'],
    })


def test_fix_outliers_clip_grades_superior(df_con_anomalias):
    """Un grade de 118 debe quedar recortado a 100, no eliminado."""
    df_resultado = fix_outliers(df_con_anomalias)

    assert df_resultado.loc[0, 'grades'] == 100.0
    assert len(df_resultado) == len(df_con_anomalias)  # No se pierde ninguna fila.


def test_fix_outliers_clip_addiction_score_negativo(df_con_anomalias):
    """Un addiction_score de -5 debe quedar recortado a 0."""
    df_resultado = fix_outliers(df_con_anomalias)

    assert df_resultado.loc[0, 'addiction_score'] == 0.0


def test_fix_outliers_no_afecta_valores_validos(df_con_anomalias):
    """Los valores ya dentro de rango no deben modificarse."""
    df_resultado = fix_outliers(df_con_anomalias)

    assert df_resultado.loc[1, 'grades'] == 45.0
    assert df_resultado.loc[2, 'addiction_score'] == 20.0


def test_fix_dtypes_stress_level_es_ordinal(df_con_anomalias):
    """stress_level debe quedar como categórica ordenada Low < Medium < High."""
    df_resultado = fix_dtypes(df_con_anomalias)

    assert df_resultado['stress_level'].dtype.ordered
    assert list(df_resultado['stress_level'].dtype.categories) == ['Low', 'Medium', 'High']


def test_engineer_features_risk_flag_umbral(df_con_anomalias):
    """risk_flag debe ser 1 cuando grades < 50, y 0 en caso contrario."""
    df_resultado = engineer_features(df_con_anomalias)

    assert df_resultado.loc[0, 'risk_flag'] == 0   # grades=118 -> sin riesgo (sin clip aún en este test)
    assert df_resultado.loc[1, 'risk_flag'] == 1   # grades=45 -> en riesgo
    assert df_resultado.loc[2, 'risk_flag'] == 0   # grades=75 -> sin riesgo


def test_engineer_features_gaming_intensity_categorias(df_con_anomalias):
    """gaming_intensity debe asignar correctamente la categoría según las horas de gaming."""
    df_resultado = engineer_features(df_con_anomalias)

    assert df_resultado.loc[0, 'gaming_intensity'] == 'Mínimo (0–1h)'   # 0.5h
    assert df_resultado.loc[1, 'gaming_intensity'] == 'Intenso (3–6h)'  # 4.0h
    assert df_resultado.loc[2, 'gaming_intensity'] == 'Extremo (6–8h)'  # 7.0h


def test_clean_data_pipeline_completo(df_con_anomalias):
    """El pipeline completo debe dejar el DataFrame sin ninguna anomalía pendiente."""
    df_resultado = clean_data(df_con_anomalias)

    assert df_resultado['grades'].max() <= 100
    assert df_resultado['addiction_score'].min() >= 0
    assert set(df_resultado['risk_flag'].unique()).issubset({0, 1})
    assert df_resultado['stress_level'].dtype.ordered