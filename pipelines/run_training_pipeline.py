"""
Pipeline de entrenamiento con tracking en MLflow.

Entrena los 3 modelos finales del proyecto GamingAcademicPerformance,
cada uno como un run independiente y comparable en la UI de MLflow.
"""
import os
import pickle
import tempfile
import mlflow

from src.data.cleaning import load_raw, clean_data, save_processed
from src.data.preprocessing import (
    encode_categoricals,
    split_classification, split_regression, prepare_cluster_data
)
from src.models.train import train_xgb_classifier, train_rf_regressor, train_kmeans

RAW_DATA_PATH = "data/raw/Gaming_Academic_Performance.csv"
PROCESSED_DATA_PATH = "data/processed/GAP_clean.csv"

def log_encoder(encoders: dict) -> None:
    """
    Serializa el diccionario de encoders a un archivo temporal
    y lo loguea como artifact en el run activo de MLflow.
    El archivo temporal se elimina después para no dejar basura en el repo.
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(encoders, tmp)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path="encoder")
    os.remove(tmp_path)

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # Indica a MLFlow dónde guardar todo (parámetros, métricas, modelos)
    mlflow.set_experiment("GAC_MLOps") # Directorio que agrupa runs relacionados.
    mlflow.autolog() # Activa el loggin automático.

    # --- Limpieza ---
    df_raw = load_raw(RAW_DATA_PATH)
    df = clean_data(df_raw)
    save_processed(df, PROCESSED_DATA_PATH)

    # --- Preprocesamiento (ahora pedimos también los encoders) ---
    df, encoders = encode_categoricals(df, return_encoders=True)

    # --- Run 1: XGBoost clasificación ---
    X_train, X_test, y_train, y_test = split_classification(df)
    with mlflow.start_run(run_name="xgb_classifier"):
        model, metrics = train_xgb_classifier(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(metrics)
        log_encoder(encoders)  # Guardamos el encoder junto al modelo.
        print(f"[XGBoost clasificador] ROC-AUC: {metrics['roc_auc']:.4f}")

    # --- Run 2: Random Forest regresión ---
    X_train, X_test, y_train, y_test = split_regression(df)
    with mlflow.start_run(run_name="rf_regressor"):
        model, metrics = train_rf_regressor(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(metrics)
        log_encoder(encoders)  # Mismo encoder, cada run es autónomo.
        print(f"[Random Forest regresor] R²: {metrics['r2']:.4f}")

    # --- Run 3: K-Means clustering ---
    X_cluster_scaled, _ = prepare_cluster_data(df)
    with mlflow.start_run(run_name="kmeans_clustering"):
        model, metrics = train_kmeans(X_cluster_scaled, k=2)
        mlflow.log_metrics(metrics)
        print(f"[K-Means] Silhouette Score: {metrics['silhouette_score']:.4f}")

    print("\nPipeline completado. Lanza 'mlflow ui --backend-store-uri sqlite:///mlflow.db' para ver los resultados.")


if __name__ == "__main__":
    main()