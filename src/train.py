import json
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.config import RANDOM_STATE, TEST_SIZE, MIN_QUALITY_SCORE, MODEL_PATH, METRICS_PATH
from src.data import load_data, validate_data

def train_model():
    print("[INFO] Iniciando proceso de entrenamiento y validación del modelo...")
    
    # 1. Cargar y validar datos
    df = load_data()
    df = validate_data(df)
    
    # 2. Separar características (X) y objetivo (y)
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    # 3. División train/test con semilla fija (Reproducibilidad)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Datos divididos: {len(X_train)} filas de entrenamiento, {len(X_test)} filas de prueba.")
    
    # 4. Entrenar el modelo (Random Forest Regressor)
    n_estimators = 100
    max_depth = 15
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluar el modelo (Corregido para scikit-learn moderno)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"[EVALUACIÓN] Métrica R2 Score: {r2:.4f}")
    print(f"[EVALUACIÓN] Métrica RMSE: {rmse:.4f}")
    
    # 6. Puerta de calidad (Quality Gate) exigida por la rúbrica
    if r2 < MIN_QUALITY_SCORE:
        raise ValueError(f"[PUERTA DE CALIDAD FALLIDA] El R2 ({r2:.4f}) es menor al mínimo ({MIN_QUALITY_SCORE}).")
    print(f"[ÉXITO] El modelo superó la puerta de calidad.")
    
    # 7. Guardar el modelo entrenado
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Modelo guardado en: {MODEL_PATH}")
    
    # 8. Guardar la ficha del modelo (models/metrics.json) requerida por la rúbrica
    metrics_data = {
        "version": "1.0.0",
        "date": datetime.datetime.now().isoformat(),
        "parameters": {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": RANDOM_STATE},
        "metrics": {"r2_score": float(r2), "rmse": float(rmse)}
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[INFO] Ficha de métricas guardada en: {METRICS_PATH}")

if __name__ == "__main__":
    train_model()