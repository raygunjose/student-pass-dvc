import pandas as pd
import joblib
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

# Load Params
params = yaml.safe_load(open("params.yaml"))
test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

# Load data
df = pd.read_csv("data/processed/students_processed.csv")

X = df[["study_hours", "attendance"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")
print("Model training completed")