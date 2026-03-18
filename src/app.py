

# building an API for the pipeline model

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load pipeline (instead of model + scaler)
pipeline = joblib.load("models/pipeline.pkl")


# Define input schema
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
        # Convert input to numpy array
        features = np.array(data.features).reshape(1, -1)

        # Direct prediction (pipeline handles scaling)
        prediction = pipeline.predict(features)[0]
        
        # Return result
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
