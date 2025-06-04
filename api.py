from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from typing import List
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="API de predicción LSTM")

# Cargar modelo al iniciar
#model = load_model("data/06_models/modelo.keras")
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_NAME = "modelo"

blob_service_client = BlobServiceClient.from_connection_string(CONN_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Descargar modelo
blob_client = container_client.get_blob_client("modelo.keras")

with open("model.keras", "wb") as f:
    download_stream = blob_client.download_blob()
    f.write(download_stream.readall())

# Cargar modelo
model = load_model("model.keras", compile=False)

# Modelo de entrada
class PredictRequest(BaseModel):
    data: List[List[List[float]]]

@app.post("/predict")
def predict(request: PredictRequest):
    arr = np.array(request.data, dtype=np.float32)
    
    # Asegurarse que la forma sea la esperada (1, 15, 10)
    if arr.shape != (1, 15, 10):
        return {"error": f"Forma de input incorrecta, se espera (1,15,10) y se recibió {arr.shape}"}
    
    # Hacer la predicción con el modelo
    prediction = model.predict(arr)
    
    # Convertir la predicción a lista para JSON serializable
    prediction_list = prediction.tolist()
    
    return {"prediction": prediction_list}