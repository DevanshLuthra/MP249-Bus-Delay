import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.mp249.config import ROOT, load_config, get_data_hash
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient

def main():
    config = load_config()

    train = pd.read_csv(ROOT / config["data"]["cleaned_train"])
    later = pd.read_csv(ROOT / config["data"]["later"])
    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)

    features = config["features"]["numerical"] + config["features"]["categorical"]
    target = config["data"]["target"]

    split = int(len(later) * config["retraining"]["training_fraction"])
    combined = pd.concat([train, later.iloc[:split]], ignore_index=True)
    check = later.iloc[split:]

    template = joblib.load(ROOT / "models/corrected_v1.joblib")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    client = MlflowClient()
    name = config["mlflow"]["registered_model_name"]

    if client.search_model_versions(f"name='{name}'"):
        raise RuntimeError("Registry is not empty. Inspect existing versions before continuing.")

    results = {}

    for label, dataset in [("v1", train), ("v2", combined)]:
        model = clone(template)
        model.fit(dataset[features], dataset[target])

        mae = mean_absolute_error(
            check[target],
            model.predict(check[features])
        )

        with mlflow.start_run(run_name=f"Final_{label}") as run:
            mlflow.log_param("data_sha256", get_data_hash())
            mlflow.log_param("model_version", label)
            mlflow.log_param("n_estimators", model.get_params()["model__n_estimators"])
            mlflow.log_param("max_depth", model.get_params()["model__max_depth"])
            mlflow.log_param("training_rows", len(dataset))
            mlflow.log_param("check_rows", len(check))
            mlflow.log_metric("mae", mae)

            info = mlflow.sklearn.log_model(
                sk_model=model,
                name=f"bus_delay_{label}",
                serialization_format="cloudpickle"
            )

            registered = mlflow.register_model(info.model_uri, name)

            joblib.dump(model, ROOT / f"models/final_{label}.joblib")

            results[label] = {
                "version": str(registered.version),
                "mae": mae,
                "run_id": run.info.run_id
            }

            print(label, results[label], flush=True)

    improvement = (
        results["v1"]["mae"] - results["v2"]["mae"]
    ) / results["v1"]["mae"]

    champion = "v2" if improvement >= config["retraining"]["minimum_mae_improvement"] else "v1"

    client.set_registered_model_alias(
        name,
        config["mlflow"]["champion_alias"],
        results[champion]["version"]
    )

    print("\nIMPROVEMENT:", improvement * 100)
    print("CHAMPION:", champion)
    print("CHAMPION VERSION:", results[champion]["version"])
    print("REGISTRATION COMPLETED")

if __name__ == "__main__":
    main()
