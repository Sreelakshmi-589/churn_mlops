import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#load model and scalar
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

#load dataset
data = pd.read_csv("data/churn.csv")

#same preprocessing
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
data = data.drop("customerID", axis=1)

data = pd.get_dummies(data)

#split feature and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

#scale using trained scalar
X_scaled = scaler.transform(X)

#predictions
preds = model.predict(X_scaled)

#metrics
acc = accuracy_score(y, preds)
f1 = f1_score(y, preds)
cm = confusion_matrix(y, preds)


#print results
print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(cm)
