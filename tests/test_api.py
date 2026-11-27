
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_prediction():
    response = client.post("/predict", json={
        "route_km": -5,
        "stops": 20,
        "passengers": 45,
        "hour": 10,
        "temp_c": 30,
        "driver_exp_yr": 8,
        "weather": "clear",
        "depot_zone": "North"
    })

    assert response.status_code == 422


def test_frontend():
    response = client.get("/app/")
    assert response.status_code == 200
    assert "html" in response.headers["content-type"]
