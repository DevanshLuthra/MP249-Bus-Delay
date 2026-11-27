import joblib
import mlflow
import pandas as pd

from src.config import ROOT, load_config, get_data_hash
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

config = load_config()

df = pd.read_csv(ROOT / config["data"]["cleaned_train"])
df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

numeric = config["features"]["numerical"]
categorical = config["features"]["categorical"]
features = numeric + categorical
target = config["data"]["target"]

split = int(len(df) * (1 - config["training"]["validation_fraction"]))

X_train = df.iloc[:split][features]
y_train = df.iloc[:split][target]
X_val = df.iloc[split:][features]
y_val = df.iloc[split:][target]

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

best_mae = float("inf")
best_run = None

for settings in config["training"]["experiments"]:
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            **settings,
            random_state=config["project"]["random_state"],
            n_jobs=-1
        ))
    ])

    with mlflow.start_run(run_name=f"RF_{settings['n_estimators']}_{settings['max_depth']}") as run:
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))

        mlflow.log_params(settings)
        mlflow.log_param("data_sha256", get_data_hash())
        mlflow.log_param("random_state", config["project"]["random_state"])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_metric("mae", mae)

        print(f"Settings: {settings} | MAE: {mae:.6f}", flush=True)

        if mae < best_mae:
            best_mae = mae
            best_run = run.info.run_id
            joblib.dump(model, ROOT / "models" / "corrected_v1.joblib")

print("\nBEST EXPERIMENT")
print("Best MAE:", best_mae)
print("Best run ID:", best_run)
print("Model saved: models/corrected_v1.joblib")
