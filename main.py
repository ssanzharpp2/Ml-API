from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

if not MODEL_PATH.exists():
    raise RuntimeError(f"File not found: {MODEL_PATH}. Train model.")
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Penguin Species ML API", version="1.0.0")
class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., gt=0, description="Bill length, mm")
    bill_depth_mm: float = Field(..., gt=0, description="Bill depth, mm")
    flipper_length_mm: float = Field(..., gt=0, description="Flipper length, mm")
    body_mass_g: float = Field(..., gt=0, description="Body mass, g")
    sex: Literal[0, 1] = Field(..., description="Sex: 0 - female, 1 - male")

    model_config = {
        "json_schema_extra": {
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "sex": 1,
            }
        }
    }


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: PenguinFeatures):
    X = pd.DataFrame([[getattr(features, name) for name in FEATURES]], columns=FEATURES)

    try:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {exc}")

    return {
        "prediction": str(prediction),
        "probabilities": {
            str(cls): round(float(p), 4) for cls, p in zip(model.classes_, probabilities)
        },
    }
