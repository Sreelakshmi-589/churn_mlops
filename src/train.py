import pandas as pd #to handle data tables

from sklearn.model_selection import train_test_split #splits data into training set and testing set
from sklearn.preprocessing import StandardScaler #to normalise values: helps nn learn better
from sklearn.neural_network import MLPClassifier #AI model

import os #manage folders
import joblib #saves model

os.makedirs("models", exist_ok=True) #creates models folder, if not present

data = pd.read_csv("data/churn.csv") #loads dataset

#data cleaning
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

#target conversion
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

#dropping unwanted columns
data = data.drop("customerID", axis=1)

#encoding categorical data
data = pd.get_dummies(data)

#split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50)

#train
model.fit(X_train, y_train)

#saving model
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Training completed")
