from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

payload = {
    "route_km": 15.5,
    "stops": 22,
    "passengers": 45,
    "hour": 14,
    "temp_c": 28.5,
    "driver_exp_yr": 12,
    "weather": "clear",
    "depot_zone": "North"
}

def test_champion_prediction():
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["model_version"] == "2"
    assert result["predicted_delay_minutes"] >= 0

def test_stops_below_minimum():
    response = client.post("/predict", json={**payload, "stops": 2.4})
    assert response.status_code == 422

def test_experience_above_maximum():
    response = client.post("/predict", json={**payload, "driver_exp_yr": 46})
    assert response.status_code == 422

def test_exact_input_boundaries():
    response = client.post("/predict", json={
        **payload,
        "stops": 2.5,
        "driver_exp_yr": 45
    })
    assert response.status_code == 200
