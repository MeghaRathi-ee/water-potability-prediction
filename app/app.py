from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(
    title="Water Potability Prediction",
    description="Predict Water Potability"
)

# Load model (relative path)
MODEL_PATH = os.path.join("models", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- Input Schema ----------
class WaterInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# ---------- Root ----------
@app.get("/")
def index():
    return {"message": "Welcome to Water Potability Prediction API"}


# ---------- Predict ----------
@app.post("/predict")
def predict(data: WaterInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_df)[0]

    return {
        "potability": int(prediction),
        "result": "Drinkable" if prediction == 1 else "Not Drinkable"
    }
