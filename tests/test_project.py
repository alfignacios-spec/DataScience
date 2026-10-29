import os
import pytest
from fastapi.testclient import TestClient
from src.config import MODEL_PATH, RAW_DATA_PATH
from src.data import load_data, validate_data
from src.predict import load_trained_model, predict_single
from src.api import app

# Inicializar el cliente de pruebas para la API de FastAPI
client = TestClient(app)

def test_data_loading_and_validation():
    """Prueba la capa de datos: verifica que existan y pasen la validación estricta."""
    assert RAW_DATA_PATH.exists(), f"[ERROR] No se encuentra el archivo de datos en {RAW_DATA_PATH}"
    df = load_data(RAW_DATA_PATH)
    df_clean = validate_data(df)
    assert len(df_clean) > 0, "[ERROR] El dataset validado está vacío."

def test_model_existence_and_prediction():
    """Prueba la capa del modelo: verifica que exista el archivo .joblib y prediga correctamente."""
    assert MODEL_PATH.exists(), f"[ERROR] El archivo del modelo no existe en {MODEL_PATH}. Ejecute src/train.py"
    model = load_trained_model()
    assert model is not None, "[ERROR] El modelo cargado es None."
    
    # Probar una predicción de prueba con un diccionario de características válidas
    sample_data = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    prediction = predict_single(sample_data)
    assert isinstance(prediction, float), "[ERROR] La predicción debe ser un número flotante."
    assert prediction > 0, "[ERROR] El precio predicho debe ser mayor a 0."

def test_api_root_and_model_info():
    """Prueba los endpoints de salud y metadatos de la API."""
    response_root = client.get("/")
    assert response_root.status_code == 200
    assert response_root.json()["status"] == "online"
    
    response_info = client.get("/model-info")
    assert response_info.status_code == 200
    assert "metrics" in response_info.json()

def test_api_predict_endpoint_valid():
    """Prueba el endpoint /predict de la API con datos válidos (espera código 200)."""
    payload = {
        "MedInc": 6.5,
        "HouseAge": 15.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.1,
        "Population": 2000.0,
        "AveOccup": 2.5,
        "Latitude": 35.0,
        "Longitude": -119.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_status"] == "success"
    assert "predicted_house_value" in data

def test_api_predict_endpoint_invalid():
    """Prueba el endpoint /predict con datos inválidos (espera código 422 de Pydantic)."""
    # Ingreso medio negativo (violates gt=0 constraint)
    invalid_payload = {
        "MedInc": -2.0,
        "HouseAge": 15.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.1,
        "Population": 2000.0,
        "AveOccup": 2.5,
        "Latitude": 35.0,
        "Longitude": -119.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, "[ERROR] La API debió rechazar el ingreso negativo con código 422."