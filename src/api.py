import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import METRICS_PATH
from src.predict import predict_single, load_trained_model

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Precios de Viviendas",
    description="Proyecto MLOps - Data Science G8",
    version="1.0.0"
)

# Definir el esquema de validación de entradas con Pydantic (Requisito de la rúbrica)
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Ingreso medio en el bloque de viviendas")
    HouseAge: float = Field(..., ge=0, le=150, description="Antigüedad media de las casas")
    AveRooms: float = Field(..., gt=0, description="Número medio de habitaciones por hogar")
    AveBedrms: float = Field(..., gt=0, description="Número medio de dormitorios por hogar")
    Population: float = Field(gt=0, description="Población total en el bloque")
    AveOccup: float = Field(..., gt=0, description="Número medio de miembros por hogar")
    Latitude: float = Field(..., ge=32.0, le=43.0, description="Latitud geográfica")
    Longitude: float = Field(..., ge=-125.0, le=-114.0, description="Longitud geográfica")

@app.get("/")
def read_root():
    """Endpoint de salud raíz."""
    return {"status": "online", "message": "API de Predicción de Precios de Viviendas funcionando correctamente."}

@app.get("/model-info")
def get_model_info():
    """Devuelve la ficha del modelo (métricas y parámetros) desde metrics.json (Requisito de la rúbrica)."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Ficha del modelo no encontrada. El modelo aún no ha sido entrenado.")
    
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    return metrics_data

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Endpoint de predicción. 
    Valida automáticamente las entradas usando Pydantic (si hay error, devuelve HTTP 422 claro).
    """
    try:
        # Verificar si el modelo existe antes de predecir
        load_trained_model()
        
        # Convertir datos de entrada a diccionario
        input_data = features.model_dump()
        
        # Obtener predicción
        predicted_price = predict_single(input_data)
        
        return {
            "prediction_status": "success",
            "predicted_house_value": predicted_price,
            "units": "Hundreds of thousands of dollars ($100k)"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no disponible en el servidor. Ejecute el entrenamiento.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))