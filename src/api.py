from pathlib import Path
from functools import lru_cache
import json
import logging
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name
        }

        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()

        return json.dumps(payload)


logger = logging.getLogger("bus_prediction")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

app = FastAPI(
    title="Surat City Bus Delay Prediction API",
    description="Predict bus delays using the champion Random Forest model.",
    version="2.0.0"
)

class BusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_km: float = Field(gt=0)
    stops: float = Field(ge=2.5)
    passengers: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    temp_c: float
    driver_exp_yr: float = Field(ge=0, le=45)
    weather: str
    depot_zone: str

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Surat Bus Delay Prediction API is running"}

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Model unavailable")

@app.post("/predict")
def predict(data: BusInput):
    try:
        model = load_model()
    except Exception:
        raise HTTPException(status_code=503, detail="Model unavailable")

    features = pd.DataFrame([data.model_dump()])
    prediction = float(model.predict(features)[0])
    prediction = round(max(0, prediction), 2)

    logger.info({
        "event": "prediction",
        "model_version": "2",
        "predicted_delay_minutes": prediction,
        "status": "success"
    })

    return {
        "predicted_delay_minutes": prediction,
        "model": "RandomForestRegressor",
        "model_version": "2"
    }

app.mount(
    "/app",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend"
)
