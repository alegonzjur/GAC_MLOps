"""
API REST con FastAPI para servir predicciones de los modelos entrenados.

Endpoints:
    POST /predict/risk   -> Clasificación (XGBoost): riesgo académico
    POST /predict/grades -> Regresión (Random Forest): predicción de nota
"""

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import StudentFeatures, RiskPredictionResponse, GradesPredictionResponse
from api.model_loader import load_model_and_encoder, build_feature_vector

xgb_model, xgb_encoders, rf_model, rf_encoders = None, None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model, xgb_encoders, rf_model, rf_encoders

    xgb_model, xgb_encoders = load_model_and_encoder(
        "xgb_classifier.pkl", "encoders_xgb.pkl"
    )
    rf_model, rf_encoders = load_model_and_encoder(
        "rf_regressor.pkl", "encoders_rf.pkl"
    )
    yield


app = FastAPI(
    title="GAC_MLOps API",
    description="Predicciones de riesgo académico y nota a partir de hábitos de gaming.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(student: StudentFeatures):
    try:
        X = np.array(build_feature_vector(student.model_dump(), xgb_encoders))
        risk_flag = int(xgb_model.predict(X)[0])
        risk_probability = float(xgb_model.predict_proba(X)[0][1])
        return RiskPredictionResponse(
            risk_flag=risk_flag,
            risk_probability=risk_probability
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/grades", response_model=GradesPredictionResponse)
def predict_grades(student: StudentFeatures):
    try:
        X = np.array(build_feature_vector(student.model_dump(), rf_encoders))
        predicted_grades = float(rf_model.predict(X)[0])
        return GradesPredictionResponse(predicted_grades=predicted_grades)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))