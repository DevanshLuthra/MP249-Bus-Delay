
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = [
    "route_km", "stops", "passengers", "hour",
    "temp_c", "driver_exp_yr"
]

CATEGORICAL_FEATURES = ["weather", "depot_zone"]


def create_sample_data():
    return pd.DataFrame([
        [10, 15, 30, 8, 28, 5, "clear", "North"],
        [15, 20, 45, 10, 30, 8, "rain", "South"],
        [20, 25, 60, 17, 32, 10, "clear", "East"],
        [12, 18, 35, 12, 29, 6, "rain", "West"]
    ], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def create_pipeline():
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=10, random_state=42))
    ])


def test_model_training():
    X = create_sample_data()
    y = [5, 10, 15, 8]

    model = create_pipeline()
    model.fit(X, y)

    assert hasattr(model.named_steps["model"], "estimators_")


def test_model_prediction():
    X = create_sample_data()
    y = [5, 10, 15, 8]

    model = create_pipeline()
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == 4
    assert np.isfinite(predictions).all()


def test_unseen_category():
    X = create_sample_data()
    y = [5, 10, 15, 8]

    model = create_pipeline()
    model.fit(X, y)

    new_data = X.iloc[[0]].copy()
    new_data["weather"] = "fog"

    prediction = model.predict(new_data)

    assert np.isfinite(prediction[0])
