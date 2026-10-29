import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_trained_model():
    """Carga el modelo entrenado desde el archivo joblib."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[ERROR] No se encontró el modelo entrenado en {MODEL_PATH}. Ejecuta primero src/train.py")
    return joblib.load(MODEL_PATH)

def predict_single(data: dict) -> float:
    """
    Recibe un diccionario con las características de una casa,
    las convierte en un DataFrame y devuelve la predicción del precio.
    """
    model = load_trained_model()
    
    # Convertir el diccionario de entrada a DataFrame de pandas con 1 sola fila
    df_input = pd.DataFrame([data])
    
    # Realizar la predicción
    prediction = model.predict(df_input)
    return float(prediction[0])