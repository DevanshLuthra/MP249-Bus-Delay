
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "later.csv"
REPORT_PATH = ROOT / "reports" / "drift_report.csv"
SUMMARY_PATH = ROOT / "reports" / "drift_summary.json"

NUMERICAL_FEATURES = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
]

CATEGORICAL_FEATURES = [
    "weather",
    "depot_zone",
]

THRESHOLD = 0.05


def numerical_drift(reference, current):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return np.nan

    return float(ks_2samp(reference, current).pvalue)


def categorical_drift(reference, current):
    reference = reference.fillna("missing").astype(str).str.strip().str.lower()
    current = current.fillna("missing").astype(str).str.strip().str.lower()

    reference_counts = reference.value_counts()
    current_counts = current.value_counts()

    categories = reference_counts.index.union(current_counts.index)

    table = np.array([
        reference_counts.reindex(categories, fill_value=0).values,
        current_counts.reindex(categories, fill_value=0).values,
    ])

    if table.shape[1] < 2:
        return 1.0

    return float(chi2_contingency(table)[1])


def main():
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    results = []

    for feature in NUMERICAL_FEATURES:
        p_value = numerical_drift(train[feature], later[feature])

        results.append({
            "feature": feature,
            "type": "numerical",
            "test": "KS",
            "p_value": p_value,
            "drift_detected": bool(np.isfinite(p_value) and p_value < THRESHOLD),
        })

    for feature in CATEGORICAL_FEATURES:
        p_value = categorical_drift(train[feature], later[feature])

        results.append({
            "feature": feature,
            "type": "categorical",
            "test": "Chi-square",
            "p_value": p_value,
            "drift_detected": bool(np.isfinite(p_value) and p_value < THRESHOLD),
        })

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(REPORT_PATH, index=False)

    drifted_features = report.loc[
        report["drift_detected"], "feature"
    ].tolist()

    summary = {
        "reference_dataset": "train_clean.csv",
        "current_dataset": "later.csv",
        "threshold": THRESHOLD,
        "total_features": len(report),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "overall_drift_detected": len(drifted_features) > 0,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nDATA DRIFT DETECTION REPORT")
    print("=" * 55)
    print(report.to_string(index=False))
    print("=" * 55)
    print(f"Drifted features: {len(drifted_features)}/{len(report)}")
    print(f"Overall drift detected: {summary['overall_drift_detected']}")
    print(f"\nCSV report saved: {REPORT_PATH}")
    print(f"JSON summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
