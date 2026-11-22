
from pathlib import Path
from datetime import date
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "bus_delay_model.joblib"

app = FastAPI(
    title="Surat City Bus Delay Prediction API",
    description="Predict bus delays using a trained Random Forest model.",
    version="1.0.0"
)


class BusInput(BaseModel):
    date: date
    route_km: float = Field(gt=0)
    stops: int = Field(gt=0)
    passengers: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    temp_c: float
    driver_exp_yr: float = Field(ge=0)
    weather: str
    depot_zone: str


@app.get("/")
def home():
    return {"message": "Surat Bus Delay Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL_PATH.exists() else "model_missing"
    }


@app.post("/predict")
def predict(data: BusInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Run training first."
        )

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([{
        "route_km": data.route_km,
        "stops": data.stops,
        "passengers": data.passengers,
        "hour": data.hour,
        "temp_c": data.temp_c,
        "driver_exp_yr": data.driver_exp_yr,
        "weather": data.weather,
        "depot_zone": data.depot_zone,
        "day_of_week": data.date.weekday(),
        "month": data.date.month
    }])

    prediction = float(model.predict(features)[0])

    return {
        "predicted_delay_minutes": round(max(0, prediction), 2),
        "model": "RandomForestRegressor",
        "model_version": "2"
    }
app.mount(
    "/app",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend"
)
