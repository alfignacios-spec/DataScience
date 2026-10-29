import pathlib

# Directorios raíz del proyecto
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Configuración del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_QUALITY_SCORE = 0.75  # Puerta de calidad (R2 Score mínimo)

# Archivos de datos
RAW_DATA_PATH = DATA_DIR / "housing.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"