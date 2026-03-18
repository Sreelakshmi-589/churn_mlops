#building an api for the model

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Define input schema using Pydantic
class InputData(BaseModel):
    features: List[float]


# Root endpoint (just to check API is running)
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input list to numpy array
        features = np.array(data.features).reshape(1, -1)

        # Scale input using trained scaler
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Return result
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
