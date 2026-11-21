
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "delay_min"])
df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

df["day_of_week"] = df["date"].dt.day_of_week
df["month"] = df["date"].dt.month

features = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
    "weather",
    "depot_zone",
    "day_of_week",
    "month",
]

target = "delay_min"

split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
val_df = df.iloc[split:].copy()

X_train = train_df[features].copy()
X_val = val_df[features].copy()
y_train = train_df[target]
y_val = val_df[target]

X_train.loc[X_train["stops"] <= 0, "stops"] = np.nan
X_val.loc[X_val["stops"] <= 0, "stops"] = np.nan

numeric_features = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
    "day_of_week",
    "month",
]

categorical_features = ["weather", "depot_zone"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
mlflow.set_experiment("MP249_Bus_Delay_Prediction")

with mlflow.start_run(run_name="RandomForest_Baseline"):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    mlflow.log_params(
        {
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "split": "chronological_80_20",
        }
    )

    mlflow.log_metrics(
        {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    joblib.dump(pipeline, MODEL_DIR / "bus_delay_model.joblib")

    mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="bus_delay_model",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree",
    ],
)

    print("Training completed successfully!")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("Model saved to:", MODEL_DIR / "bus_delay_model.joblib")
