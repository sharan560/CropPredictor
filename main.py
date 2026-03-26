from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# =========================================
# LOAD MODEL
# =========================================
model = CatBoostClassifier()
model.load_model("crop_model.cbm")

app = FastAPI()

# =========================================
# INPUT SCHEMA
# =========================================
class CropInput(BaseModel):
    soil_type: str
    soil_temp: float
    env_temp: float
    moisture: float
    ph: Optional[float] = None
    rainfall: Optional[float] = None

# =========================================
# ROOT API
# =========================================
@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🚀"}

# =========================================
# PREDICTION API
# =========================================
@app.post("/predict")
def predict(data: CropInput):

    rainfall_value = 50.0 if data.rainfall is None else data.rainfall
    temp_diff = data.env_temp - data.soil_temp
    water_index = data.moisture + rainfall_value
    ph_value = 6.5 if data.ph is None else data.ph

    sample = pd.DataFrame({
        'soil_type':[data.soil_type],
        'soil_temp':[data.soil_temp],
        'env_temp':[data.env_temp],
        'moisture':[data.moisture],
        'ph':[ph_value],
        'rainfall':[rainfall_value],
        'temp_diff':[temp_diff],
        'water_index':[water_index]
    })

    probs = model.predict_proba(sample)[0]
    crops = model.classes_

    top5_idx = np.argsort(probs)[-5:][::-1]

    result = []
    for i in top5_idx:
        result.append({
            "crop": str(crops[i]),
            "suitability": float(probs[i] * 100)
        })

    return {
        "top_predictions": result
    }