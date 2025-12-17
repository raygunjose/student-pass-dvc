from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Student Pass Prediction API")

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API is running"}

@app.post("/predict")
def predict(study_hours: float, attendance: float):
    X = np.array([[study_hours, attendance]])
    prediction = model.predict(X)[0]

    return {
        "study_hours": study_hours,
        "attendance": attendance,
        "prediction": "Pass" if prediction == 1 else "Fail"
    }
