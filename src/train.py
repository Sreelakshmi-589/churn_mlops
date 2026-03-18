
import pandas as pd #to handle data tables

from sklearn.model_selection import train_test_split #splits data into training set and test set
from sklearn.preprocessing import StandardScaler #to normalise values: helps nn learn better
from sklearn.neural_network import MLPClassifier #AI model
from sklearn.pipeline import Pipeline 

import os #manage folders
import joblib #saves model

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/churn.csv")

# Data cleaning
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

# Target conversion
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Drop unnecessary column
data = data.drop("customerID", axis=1)

# Encode categorical variables
data = pd.get_dummies(data)

# Split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split (IMPORTANT: no scaling before this now)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline (scaler + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "models/pipeline.pkl")

print("Pipeline training completed")
