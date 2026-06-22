"""
Exporta los modelos entrenados y sus encoders desde MLflow a disco.

Genera en models/:
    - xgb_classifier.pkl
    - rf_regressor.pkl
    - encoders_xgb.pkl
    - encoders_rf.pkl

Esto desacopla la API de MLflow en producción: el contenedor Docker
carga directamente desde disco, sin depender del tracking server.
"""

import os
import pickle
import joblib
import mlflow.xgboost
import mlflow.sklearn
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODELS_DIR = "models"

# --- IDs de las runs ---
XGB_RUN_ID = "57aec008c1754be4aa81e18249bb92d9"
RF_RUN_ID = "75557c2ac5ea481e94efb17cc075daf1"

def export_model(run_id: str, flavor, output_filename: str) -> None:
    """Carga un modelo desde MLflow y lo guarda en models/ con joblib."""
    model_uri = f"runs:/{run_id}/model"
    model = flavor.load_model(model_uri)
    output_path = os.path.join(MODELS_DIR, output_filename)
    joblib.dump(model, output_path)
    print(f"Modelo exportado -> {output_path}")


def export_encoder(run_id: str, output_filename: str) -> None:
    """Descarga el encoder desde MLflow artifacts y lo guarda en models/."""
    client = MlflowClient()
    encoder_path = client.download_artifacts(run_id, "encoder")
    pkl_file = [f for f in os.listdir(encoder_path) if f.endswith(".pkl")][0]

    with open(os.path.join(encoder_path, pkl_file), "rb") as f:
        encoders = pickle.load(f)

    output_path = os.path.join(MODELS_DIR, output_filename)
    with open(output_path, "wb") as f:
        pickle.dump(encoders, f)

    print(f"Encoder exportado -> {output_path}")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Exportando XGBoost classifier...")
    export_model(XGB_RUN_ID, mlflow.xgboost, "xgb_classifier.pkl")
    export_encoder(XGB_RUN_ID, "encoders_xgb.pkl")

    print("Exportando Random Forest regressor...")
    export_model(RF_RUN_ID, mlflow.sklearn, "rf_regressor.pkl")
    export_encoder(RF_RUN_ID, "encoders_rf.pkl")

    print("\nExportación completada. Archivos en models/:")
    for f in os.listdir(MODELS_DIR):
        size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()