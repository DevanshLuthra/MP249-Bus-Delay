
from pathlib import Path
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "later.csv"
DRIFT_PATH = ROOT / "reports" / "drift_summary.json"
MODEL_PATH = ROOT / "models" / "bus_delay_model.joblib"
CANDIDATE_PATH = ROOT / "models" / "bus_delay_candidate.joblib"
REPORT_PATH = ROOT / "reports" / "retraining_report.json"

HOLDOUT_FRACTION = 0.20


def prepare_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "delay_min"])
    df = df.drop_duplicates()
    df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month

    df.loc[df["stops"] <= 0, "stops"] = np.nan

    return df


def evaluate(model, x, y):
    predictions = model.predict(x)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
    }


def main():
    with open(DRIFT_PATH, encoding="utf-8") as file:
        drift = json.load(file)

    if not drift["overall_drift_detected"]:
        print("No drift detected. Retraining skipped.")
        return

    train = prepare_data(pd.read_csv(TRAIN_PATH))
    later = prepare_data(pd.read_csv(LATER_PATH))


    split = int(len(later) * (1 - HOLDOUT_FRACTION))
    holdout_start_date = later.iloc[split]["date"]

    new_train = later[
        later["date"] < holdout_start_date
    ].copy()

    holdout = later[
        later["date"] >= holdout_start_date
    ].copy()

    if new_train.empty or holdout.empty:
        raise ValueError("Invalid chronological split")

    if new_train["date"].max() >= holdout["date"].min():
        raise ValueError("Training and holdout dates overlap")

    combined = pd.concat([train, new_train], ignore_index=True)

    existing_model = joblib.load(MODEL_PATH)
    candidate_model = clone(existing_model)

    features = list(existing_model.feature_names_in_)

    x_train = combined[features]
    y_train = combined["delay_min"]

    x_test = holdout[features]
    y_test = holdout["delay_min"]

    existing_metrics = evaluate(existing_model, x_test, y_test)

    candidate_model.fit(x_train, y_train)
    candidate_metrics = evaluate(candidate_model, x_test, y_test)

    improved = candidate_metrics["mae"] < existing_metrics["mae"]

    mlflow.set_tracking_uri(
        "sqlite:///" + (ROOT / "mlflow.db").as_posix()
    )
    mlflow.set_experiment("MP249_Bus_Delay_Prediction")

    with mlflow.start_run(run_name="Drift_Triggered_Retraining"):
        mlflow.log_params({
            "trigger": "data_drift",
            "training_rows": len(combined),
            "holdout_rows": len(holdout),
            "holdout_fraction": HOLDOUT_FRACTION,
            "model": "RandomForestRegressor",
        })

        mlflow.log_metrics({
            "existing_mae": existing_metrics["mae"],
            "candidate_mae": candidate_metrics["mae"],
            "existing_rmse": existing_metrics["rmse"],
            "candidate_rmse": candidate_metrics["rmse"],
            "existing_r2": existing_metrics["r2"],
            "candidate_r2": candidate_metrics["r2"],
        })


        mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="retrained_candidate",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )


    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, CANDIDATE_PATH)

    report = {
        "drift_triggered": True,
        "training_rows": len(combined),
        "holdout_rows": len(holdout),
        "existing_model": existing_metrics,
        "candidate_model": candidate_metrics,
        "candidate_improved": improved,
        "production_model_replaced": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nAUTOMATED RETRAINING RESULTS")
    print("=" * 45)
    print("Training rows:", len(combined))
    print("Holdout rows:", len(holdout))
    print("\nExisting model:", existing_metrics)
    print("Candidate model:", candidate_metrics)
    print("\nCandidate improved:", improved)
    print("Production model replaced: False")
    print("Report saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
