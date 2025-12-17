import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/students_processed.csv")
X = df[["study_hours", "attendance"]]
y = df["pass"]

model = joblib.load("models/model.pkl")
preds = model.predict(X)

accuracy = accuracy_score(y, preds)

metrics = {
    "accuracy":accuracy
}

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model evaluation completed")