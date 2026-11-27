import joblib
import mlflow
import pandas as pd

from src.config import ROOT, load_config
from mlflow.tracking import MlflowClient

def main():
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        config["mlflow"]["registered_model_name"],
        config["mlflow"]["champion_alias"]
    )

    version = str(champion.version)

    model = joblib.load(ROOT / config["output"]["model"])

    holdout = pd.read_csv(ROOT / config["data"]["holdout"])

    features = (
        config["features"]["numerical"]
        + config["features"]["categorical"]
    )

    predictions = model.predict(holdout[features])

    result = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions
    })

    assert len(result) == 250
    assert result["record_id"].is_unique
    assert not result.isna().any().any()

    output = ROOT / config["output"]["predictions"]
    output.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output, index=False)

    print("CHAMPION VERSION:", version)
    print("TOTAL PREDICTIONS:", len(result))
    print("COLUMNS:", list(result.columns))
    print("MISSING VALUES:", result.isna().sum().sum())
    print("DUPLICATE IDS:", result["record_id"].duplicated().sum())
    print("MIN PREDICTION:", result["prediction"].min())
    print("MAX PREDICTION:", result["prediction"].max())
    print("\nFIRST 5 PREDICTIONS:")
    print(result.head().to_string(index=False))
    print("\nSAVED:", output)

if __name__ == "__main__":
    main()
