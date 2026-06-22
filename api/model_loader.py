"""
Carga de modelos y encoders desde disco (archivos serializados).

En producción (Docker) los modelos viven en models/, exportados
previamente desde MLflow con scripts/export_models.py.
Esto desacopla la API del tracking server de MLflow.
"""

import os
import pickle
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model_and_encoder(model_filename: str, encoder_filename: str):
    """
    Carga un modelo y su encoder desde disco.

    Args:
        model_filename: nombre del archivo .pkl del modelo en models/.
        encoder_filename: nombre del archivo .pkl del encoder en models/.

    Returns:
        model: modelo entrenado listo para predict().
        encoders: diccionario de LabelEncoders ajustados en entrenamiento.
    """
    model_path = os.path.join(MODELS_DIR, model_filename)
    encoder_path = os.path.join(MODELS_DIR, encoder_filename)

    model = joblib.load(model_path)

    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


def build_feature_vector(data: dict, encoders: dict) -> list:
    """
    Convierte el input crudo de la API en el vector numérico
    que esperan los modelos, aplicando los encoders del entrenamiento.
    """
    return [[
        data['age'],
        data['gaming_hours'],
        data['study_hours'],
        data['sleep_hours'],
        data['attendance'],
        data['social_activity'],
        data['device_usage'],
        data['reaction_time_ms'],
        data['addiction_score'],
        encoders['gender'].transform([data['gender']])[0],
        encoders['gaming_genre'].transform([data['gaming_genre']])[0],
        encoders['stress_level'].transform([data['stress_level']])[0],
    ]]