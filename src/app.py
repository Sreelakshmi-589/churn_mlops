
# building an API for the pipeline model

#from fastapi import FastAPI
#from pydantic import BaseModel
#from typing import List
#import joblib
#import numpy as np

# Create FastAPI app
#app = FastAPI()

# Load pipeline (instead of model + scaler)
#pipeline = joblib.load("models/pipeline.pkl")


# Define input schema
#class InputData(BaseModel):
#    features: List[float]


# Root endpoint (just to check API is running)
#@app.get("/")
#def home():
#    return {"message": "Churn Prediction API is running"}


# Prediction endpoint
#@app.post("/predict")
#def predict(data: InputData):
#    try:
        # Convert input to numpy array
#        features = np.array(data.features).reshape(1, -1)

        # Direct prediction (pipeline handles scaling)
#        prediction = pipeline.predict(features)[0]
        
        # Return result
#        return {"prediction": int(prediction)}

#    except Exception as e:
#        return {"error": str(e)}



# building an API for the pipeline model (FULL INPUT VERSION)

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI()

# Load pipeline and columns
pipeline = joblib.load("models/pipeline.pkl")
columns = joblib.load("models/columns.pkl")


# Define FULL input schema
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Root endpoint
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to dict
        input_dict = data.dict()

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # Apply encoding (same as training)
        df = pd.get_dummies(df)

        # Align with training columns
        df = df.reindex(columns=columns, fill_value=0)

        # Predict
        prediction = pipeline.predict(df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}



