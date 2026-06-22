"""
Carga de modelos y encoders desde MLflow.

Separado de main.py para que la lógica de "conectar con MLflow y
cargar artefactos" no esté mezclada con la lógica de "servir endpoints".
"""

import os
import pickle
import mlflow.sklearn
import mlflow.xgboost

TRACKING_URI = "sqlite:///mlflow.db"


def load_model_and_encoder(run_id: str, flavor: str = "sklearn"):
    """
    Carga el modelo y el encoder del run especificado.

    Args:
        run_id: ID del run de MLflow.
        flavor: 'sklearn' para Random Forest, 'xgboost' para XGBClassifier.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"runs:/{run_id}/model"

    if flavor == "xgboost":
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)

    client = mlflow.tracking.MlflowClient()
    encoder_path = client.download_artifacts(run_id, "encoder")

    pkl_file = [f for f in os.listdir(encoder_path) if f.endswith(".pkl")][0]
    with open(os.path.join(encoder_path, pkl_file), "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


def build_feature_vector(data: dict, encoders: dict) -> list:
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