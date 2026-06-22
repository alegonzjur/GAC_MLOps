"""
Esquemas de validación de inputs y outputs de la API.

FastAPI usa estos esquemas (basados en Pydantic) para:
- Validar automáticamente que el request tiene todos los campos necesarios.
- Devolver errores claros si falta algún campo o el tipo es incorrecto.
- Generar la documentación automática en /docs.
"""

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    age: int = Field(..., ge=16, le=24, description="Edad del estudiante")
    gaming_hours: float = Field(..., ge=0, le=8)
    study_hours: float = Field(..., ge=0, le=10)
    sleep_hours: float = Field(..., ge=0, le=12)
    attendance: float = Field(..., ge=0, le=100)
    social_activity: float = Field(..., ge=0, le=5)
    device_usage: float = Field(..., ge=0, le=24)
    reaction_time_ms: float = Field(..., ge=100, le=600)
    addiction_score: float = Field(..., ge=0, le=30)
    gender: str = Field(..., description="'Male', 'Female' o 'Other'")
    gaming_genre: str = Field(..., description="'Casual', 'FPS' o 'RPG'")
    stress_level: str = Field(..., description="'Low', 'Medium' o 'High'")


class RiskPredictionResponse(BaseModel):
    risk_flag: int = Field(..., description="0 = sin riesgo, 1 = en riesgo académico")
    risk_probability: float = Field(..., description="Probabilidad de estar en riesgo (0-1)")


class GradesPredictionResponse(BaseModel):
    predicted_grades: float = Field(..., description="Nota académica predicha (0-100)")