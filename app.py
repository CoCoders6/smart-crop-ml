from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

os.makedirs("models", exist_ok=True)
model = joblib.load("models/crop_recommender.pkl")

class CropRequest(BaseModel):
    soilPh: float
    N: float = 120
    rain: float = 0
    temperature: float = 30.0
    humidity: float = 50.0

@app.post("/recommend")
async def recommend(req: CropRequest):
    pH = req.soilPh
    N = req.N
    rainfall = req.rain
    temperature = req.temperature
    humidity = req.humidity

    features = np.array([[N, temperature, humidity, pH, rainfall]])

    try:
        crop_prediction = model.predict(features)[0]
    except Exception as e:
        return {"error": str(e)}

    return {
        "input": {
            "pH": pH,
            "N": N,
            "rain": rainfall,
            "temperature": temperature,
            "humidity": humidity
        },
        "recommended_crop": str(crop_prediction)
    }
