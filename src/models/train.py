"""
Entrenamiento de los 3 modelos finales seleccionados en el notebook original:
- XGBoost (clasificación: riesgo académico)
- Random Forest (regresión: predicción de nota)
- K-Means (clustering: segmentación de perfiles)

Cada función entrena un modelo y devuelve el modelo entrenado junto con
sus métricas de evaluación en test.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.cluster import KMeans
from xgboost import XGBClassifier


def train_xgb_classifier(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena el clasificador XGBoost y calcula sus métricas en test."""
    model = XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    reporte = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'accuracy': reporte['accuracy'],
        'precision': reporte['1']['precision'],
        'recall': reporte['1']['recall'],
        'f1_score': reporte['1']['f1-score'],
    }
    return model, metrics


def train_rf_regressor(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena el regresor Random Forest y calcula sus métricas en test."""
    model = RandomForestRegressor(random_state=random_state, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    return model, metrics


def train_kmeans(X_cluster_scaled, k=2, random_state=42):
    """Entrena K-Means y calcula sus métricas de calidad de clustering."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_cluster_scaled)

    metrics = {
        'inertia': model.inertia_,
        'silhouette_score': silhouette_score(X_cluster_scaled, labels),
        'n_clusters': k,
    }
    return model, metrics