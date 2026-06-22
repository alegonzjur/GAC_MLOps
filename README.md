# GAC_MLOps

Proyecto que aplica un ciclo de vida de ML completo y reproducible sobre el proyecto base [**GamingAcademicPerformance**](https://github.com/alegonzjur/GamingAcademicPerformance), que analiza el impacto del gaming en el rendimiento académico de 8.000 estudiantes.
 
El objetivo no es solo entrenar modelos, sino demostrar ingeniería de ML aplicada: estructura modular, tracking de experimentos, tests unitarios, versionado de datos y despliegue en contenedor.
 
---

## Stack tecnológico
 
| Herramienta | Rol en el proyecto |
|---|---|
| **MLflow** | Tracking de experimentos, model registry local |
| **DVC** | Versionado de datos (raw y processed) |
| **pytest** | Tests unitarios del pipeline de datos |
| **FastAPI** | API REST para servir predicciones |
| **Docker** | Empaquetado y despliegue de la API |
| **scikit-learn 1.7.2** | Random Forest, preprocesamiento |
| **XGBoost 3.2.0** | Clasificador de riesgo académico |
| **conda** | Gestión del entorno |
 
---

## Estructura del repositorio

```
GAC_MLOps/
├── data/
│   ├── raw/              # Dataset original de Kaggle (versionado con DVC)
│   └── processed/        # Dataset limpio generado por el pipeline (DVC)
│
├── src/
│   ├── data/
│   │   ├── cleaning.py       # Limpieza de datos y feature engineering
│   │   └── preprocessing.py  # Codificación, splits, escalado
│   └── models/
│       └── train.py          # Entrenamiento de los 3 modelos finales
│
├── pipelines/
│   └── run_training_pipeline.py  # Orquestador: limpieza → preprocesamiento → MLflow
│
├── api/
│   ├── main.py           # Aplicación FastAPI con 2 endpoints de predicción
│   ├── model_loader.py   # Carga de modelos y encoders desde disco
│   ├── schemas.py        # Validación de inputs/outputs con Pydantic
│   └── __init__.py
│
├── tests/
│   └── unit/
│       ├── test_cleaning.py      # 7 tests sobre limpieza de datos
│       └── test_preprocessing.py # 9 tests sobre preprocesamiento
│
├── scripts/
│   └── export_models.py  # Exporta modelos desde MLflow a disco (para Docker)
│
├── models/               # Modelos y encoders serializados (generados por export_models.py)
│
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
│
├── notebooks/            # Exploración puntual (no lógica de producción)
├── configs/
├── pytest.ini
├── environment.yml
└── .dvcignore
```

---

## Modelos entrenados
 
| Modelo | Tarea | Target | Métrica principal |
|---|---|---|---|
| XGBoost Classifier | Clasificación binaria | `risk_flag` (riesgo académico) | ROC-AUC: **0.979** |
| Random Forest Regressor | Regresión | `grades` (nota académica) | R²: **0.92** |
| K-Means | Clustering | — | Silhouette Score trackeado |
 
---

## Instalación y uso

### 1. Clonar el repositorio
 
```bash
git clone https://github.com/alegonzjur/GAC_MLOps.git
cd GAC_MLOps
```
 
### 2. Crear el entorno conda
 
```bash
conda env create -f environment.yml
conda activate gac-mlops
```
 
### 3. Recuperar los datos (DVC)
 
```bash
dvc pull
```
 
### 4. Ejecutar el pipeline de entrenamiento
 
```bash
python -m pipelines.run_training_pipeline
```
 
Esto ejecuta en secuencia: limpieza de datos → preprocesamiento → entrenamiento de los 3 modelos → tracking en MLflow.
 
### 5. Ver los experimentos en MLflow
 
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
 
Abre `http://127.0.0.1:5000` para ver los runs, métricas, parámetros y modelos serializados.
 
### 6. Ejecutar los tests
 
```bash
pytest -v
```
 
16 tests unitarios cubriendo limpieza de datos y preprocesamiento.
 
---

## API REST — uso local
 
### Arrancar la API
 
```bash
uvicorn api.main:app --reload
```
 
Documentación interactiva disponible en `http://127.0.0.1:8000/docs`.
 
### Endpoints
 
#### `GET /health`
Comprueba que la API está activa.
 
```json
{"status": "ok"}
```
 
#### `POST /predict/risk`
Predice si un estudiante está en riesgo académico.
 
**Request:**
```json
{
  "age": 20,
  "gaming_hours": 6.5,
  "study_hours": 1.5,
  "sleep_hours": 5.0,
  "attendance": 55.0,
  "social_activity": 2.0,
  "device_usage": 10.0,
  "reaction_time_ms": 450.0,
  "addiction_score": 22.0,
  "gender": "Male",
  "gaming_genre": "FPS",
  "stress_level": "High"
}
```
 
**Response:**
```json
{
  "risk_flag": 1,
  "risk_probability": 0.97
}
```
 
#### `POST /predict/grades`
Predice la nota académica esperada.
 
**Response:**
```json
{
  "predicted_grades": 21.3
}
```
 
**Valores válidos para campos categóricos:**
- `gender`: `"Male"`, `"Female"`, `"Other"`
- `gaming_genre`: `"Casual"`, `"FPS"`, `"RPG"`
- `stress_level`: `"Low"`, `"Medium"`, `"High"`

---

## Despliegue con Docker
 
### Antes de construir la imagen — exportar los modelos
 
Los modelos deben estar serializados en `models/` antes de construir la imagen. Si acabas de ejecutar el pipeline, lanza primero:
 
```bash
python scripts/export_models.py
```
 
Esto genera en `models/`:
- `xgb_classifier.pkl`
- `rf_regressor.pkl`
- `encoders_xgb.pkl`
- `encoders_rf.pkl`
### Construir la imagen
 
```bash
docker build -f docker/Dockerfile -t gac-mlops-api:v1 .
```
 
### Ejecutar el contenedor
 
```bash
docker run -p 8000:8000 gac-mlops-api:v1
```
 
La API quedará disponible en `http://127.0.0.1:8000`. La imagen es completamente autónoma — no requiere MLflow, conda ni ninguna dependencia externa.
 
---

## Decisiones de diseño
 
**¿Por qué separar `cleaning.py` de `preprocessing.py`?**
Tienen responsabilidades distintas. `cleaning.py` corrige el dataset crudo (outliers, tipos, features derivadas) y es independiente del modelado. `preprocessing.py` prepara los datos para los modelos concretos (codificación, splits, escalado). Esto permite testear cada pieza de forma aislada y reutilizarlas en contextos distintos.
 
**¿Por qué los encoders se guardan como artifacts en MLflow?**
Sin guardar el encoder junto al modelo, la API no puede replicar exactamente la misma transformación de categóricas que vieron los modelos en entrenamiento. Guardarlos en MLflow garantiza que cada run es autónomo y reproducible: modelo + encoder + métricas + parámetros, todo trazable.
 
**¿Por qué la API carga desde disco en vez de desde MLflow?**
En producción, la API no debería depender de un servidor de tracking de experimentos. Los modelos se exportan una vez desde MLflow a archivos serializados, y Docker los empaqueta. Esto desacopla el ciclo de experimentación del ciclo de despliegue.
 
**¿Por qué DVC con remote local en vez de S3?**
El proyecto está diseñado para ejecutarse completamente en local sin servicios de pago. El remote local de DVC es conceptualmente idéntico a un remote en S3 o GCS — cambiar de uno a otro solo requiere modificar la URL del remote en `.dvc/config`, sin tocar el código.
 
**¿Por qué conda en vez de venv?**
El proyecto mezcla librerías con dependencias binarias (XGBoost, scikit-learn) donde conda ofrece mejor resolución de dependencias en Windows. En un entorno de equipo o CI/CD basado en Linux, venv con `pip-tools` sería igualmente válido.
 
---

## Errores encontrados durante el desarrollo
 
Esta sección documenta los errores reales encontrados durante la construcción del proyecto, con su causa y solución. Se incluyen como parte del proceso de aprendizaje.
 
**MLflow: filesystem backend en modo mantenimiento**
Al usar `mlflow.set_tracking_uri("file:./mlruns")`, las versiones recientes de MLflow lanzan un error indicando que el backend de ficheros está deprecado. Solución: migrar a backend SQLite con `mlflow.set_tracking_uri("sqlite:///mlflow.db")` y lanzar la UI con `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
 
**DVC: conflicto con archivos ya trackeados por git**
Al intentar añadir el CSV a DVC con `dvc add`, el comando falló porque el archivo ya estaba en git. Solución: eliminarlo del índice de git con `git rm --cached` (sin borrar el archivo de disco) antes de añadirlo a DVC.
 
**FastAPI: `@app.on_event("startup")` deprecado**
Las versiones recientes de FastAPI deprecan este decorador. Solución: migrar al patrón moderno con `@asynccontextmanager` y `lifespan`.
 
**LabelEncoder: incompatibilidad de tipos `Categorical` vs `str`**
Cuando pandas convierte una columna a dtype `category` y después `LabelEncoder` hace `fit_transform`, guarda sus clases como objetos `Categorical`, no como strings. Al intentar `transform(['M'])` desde la API con un string plano, el encoder lanzaba `"previously unseen labels: 'M'"`. Solución: añadir `.astype(str)` antes de cada `fit_transform` en `encode_categoricals()`.
 
**Docker: `KeyError: 'sklearn'` al cargar XGBClassifier**
`mlflow.sklearn.load_model()` fallaba porque MLflow había serializado el XGBClassifier con el flavor `xgboost`, no `sklearn`. Solución: cargar XGBoost con `mlflow.xgboost.load_model()` y Random Forest con `mlflow.sklearn.load_model()`.
 
**Docker: `Not supported type for data: DMatrix`**
Al intentar pasar un `xgb.DMatrix` al modelo cargado con `mlflow.xgboost`, el modelo lo rechazaba porque MLflow lo cargó como `XGBClassifier` (API sklearn), no como `Booster` nativo. Un `XGBClassifier` acepta arrays de numpy directamente y sí tiene `predict_proba()`. Solución: usar `np.array(X)` directamente en vez de envolver en `DMatrix`.
 
---
 
## Proyecto base
 
Este proyecto extiende [GamingAcademicPerformance](https://github.com/alegonzjur/GamingAcademicPerformance), que incluye el EDA, la limpieza de datos original y la exploración de modelos en notebooks.

---

## Autor

Alejandro González Jurado
[https://github.com/alegonzjur](https://github.com/alegonzjur)
