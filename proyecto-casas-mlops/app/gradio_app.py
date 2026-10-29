import gradio as gr
import requests

# URL del endpoint de predicción de nuestra API de FastAPI
API_URL = "http://127.0.0.1:8000/predict"

def predecir_precio_casa(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    """
    Función que toma los valores de los sliders de la interfaz,
    los empaqueta en JSON, los envía a la API por HTTP y devuelve el resultado.
    """
    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    try:
        # Enviar petición POST a la API de FastAPI
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            precio = data.get("predicted_house_value")
            unidades = data.get("units", "")
            return f"🏡 Precio estimado de la vivienda: ${precio:.2f} ({unidades})"
        else:
            return f"❌ Error en la API: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Error crítico: No se pudo conectar con la API. ¿Está el servidor de FastAPI encendido?"
    except Exception as e:
        return f"❌ Ocurrió un error inesperado: {str(e)}"

# Construir la interfaz visual con Gradio
demo = gr.Interface(
    fn=predecir_precio_casa,
    inputs=[
        gr.Slider(minimum=0.5, maximum=15.0, value=5.0, step=0.1, label="Ingreso Medio (MedInc)"),
        gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Antigüedad de la Casa (HouseAge)"),
        gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.1, label="Habitaciones Medias (AveRooms)"),
        gr.Slider(minimum=0.5, maximum=4.0, value=1.0, step=0.1, label="Dormitorios Medios (AveBedrms)"),
        gr.Slider(minimum=100, maximum=10000, value=1500, step=50, label="Población del Bloque (Population)"),
        gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="Ocupación Media por Hogar (AveOccup)"),
        gr.Slider(minimum=32.0, maximum=42.0, value=34.0, step=0.01, label="Latitud (Latitude)"),
        gr.Slider(minimum=-124.0, maximum=-114.0, value=-118.0, step=0.01, label="Longitud (Longitude)")
    ],
    outputs=gr.Textbox(label="Resultado de la Predicción"),
    title="Predctor de Precios de Viviendas (MLOps Project)",
    description="Desliza los valores de las características de la vivienda y haz clic en 'Submit' para consultar el precio estimado a través de la API."
)

if __name__ == "__main__":
    demo.launch()