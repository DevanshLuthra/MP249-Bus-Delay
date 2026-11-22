import hashlib
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

def run_script(script):
    return subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        capture_output=True,
        text=True
    )

def write_log(message):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "drift_monitor.log", "a", encoding="utf-8") as file:
        file.write(f"\n[{datetime.now().astimezone().isoformat()}] {message}\n")

def get_dataset_fingerprint():
    digest = hashlib.sha256()

    for path in [
        ROOT / "data" / "train_clean.csv",
        ROOT / "data" / "later.csv",
    ]:
        digest.update(path.name.encode("utf-8"))
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

    return digest.hexdigest()

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    result = run_script("src/detect_drift.py")

    write_log(result.stdout)
    write_log(result.stderr)

    if result.returncode != 0:
        print("Drift detection failed.")
        sys.exit(result.returncode)

    with open(REPORTS / "drift_summary.json", encoding="utf-8") as file:
        summary = json.load(file)

    drift = summary["overall_drift_detected"]
    state_path = REPORTS / "retraining_state.json"
    current_fingerprint = get_dataset_fingerprint()

    history_path = REPORTS / "monitoring_history.csv"
    new_file = not history_path.exists() or history_path.stat().st_size == 0

    with open(history_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["timestamp", "drift_detected", "drifted_features"])
        writer.writerow([
            timestamp,
            drift,
            ", ".join(summary["drifted_features"])
        ])

    if not drift:
        print("No drift detected. Retraining skipped.")
        return

    if state_path.exists():
        with open(state_path, encoding="utf-8") as file:
            state = json.load(file)

        if state.get("last_successful_fingerprint") == current_fingerprint:
            print("Dataset already evaluated. Retraining skipped.")
            return

    alert = (
        f"ALERT: Drift detected in "
        f"{', '.join(summary['drifted_features'])}"
    )

    with open(LOGS / "drift_alerts.log", "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {alert}\n")

    print(alert)
    print("Starting candidate model evaluation...")

    retraining = run_script("src/retrain_model.py")

    write_log(retraining.stdout)
    write_log(retraining.stderr)

    if retraining.returncode != 0:
        print("Retraining evaluation failed. Check logs.")
        sys.exit(retraining.returncode)

    report_path = REPORTS / "retraining_report.json"

    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)

    if report["candidate_improved"]:
        print("Candidate model improved.")
        print("Manual approval required before production deployment.")
    else:
        print("Candidate model did not improve. Promotion rejected.")

    print("Production model remains unchanged.")
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump({
            "last_successful_fingerprint": current_fingerprint,
            "last_evaluation": timestamp
        }, file, indent=4)
    print("Monitoring completed successfully.")

if __name__ == "__main__":
    main()
