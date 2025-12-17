import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/students.csv")

df = df.dropna()

df.to_csv("data/processed/students_processed.csv", index=False)

print("Data Preparation completed")