import json
import numpy as np
import pandas as pd

from src.mp249.config import ROOT, load_config

def calculate_psi(reference, current, bins=10, epsilon=0.0001):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    edges = np.linspace(reference.min(), reference.max(), bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)

    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)

    return float(np.sum((actual - expected) * np.log(actual / expected)))

def main():
    config = load_config()

    train = pd.read_csv(ROOT / config["data"]["cleaned_train"])
    later = pd.read_csv(ROOT / config["data"]["later"])

    settings = config["monitoring"]
    results = {}

    for feature in config["features"]["numerical"]:
        results[feature] = calculate_psi(
            train[feature].dropna(),
            later[feature].dropna(),
            settings["psi_bins"],
            settings["psi_epsilon"]
        )

    most_drifted = max(results, key=results.get)

    report = {
        "psi": results,
        "most_drifted_feature": most_drifted,
        "threshold": settings["psi_threshold"],
        "drift_detected": any(
            value >= settings["psi_threshold"]
            for value in results.values()
        )
    }

    output = ROOT / "reports/psi_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("PSI DRIFT REPORT\n")

    for feature, value in results.items():
        status = "DRIFT" if value >= settings["psi_threshold"] else "STABLE"
        print(f"{feature}: {value:.6f} | {status}")

    print("\nMOST DRIFTED FEATURE:", most_drifted)
    print("DRIFT DETECTED:", report["drift_detected"])
    print("REPORT SAVED:", output)

if __name__ == "__main__":
    main()
