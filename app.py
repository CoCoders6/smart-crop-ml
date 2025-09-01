from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/crop_recommender.pkl")

class CropRequest(BaseModel):
    soilPh: float
    N: float = 120
    rain: float = 200

@app.post("/recommend")
async def recommend(req: CropRequest):
    # Extract features
    pH = req.soilPh
    N = req.N
    rain = req.rain

    features = np.array([[pH, N, rain]])
    
    try:
        crop_prediction = model.predict(features)[0]
    except Exception as e:
        return {"error": str(e)}

    return {
        "input": {"pH": pH, "N": N, "rain": rain},
        "recommended_crop": str(crop_prediction)  # convert to string if needed
    }
