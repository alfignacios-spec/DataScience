# Usar una imagen oficial y ligera de Python
FROM python:3.11-slim

# Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de configuración y dependencias
COPY pyproject.toml ./

# Copiar las carpetas clave del proyecto (código, modelo entrenado y datos)
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Instalar las dependencias del proyecto directamente dentro del contenedor
RUN pip install --no-cache-dir .

# Exponer el puerto 8000 para que la API sea accesible
EXPOSE 8000

# Comando por defecto para encender el servidor de la API al iniciar el contenedor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]