import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    log_path = LOGS / "drift_monitor.log"

    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "detect_drift.py")],
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{timestamp}] Drift monitoring started\n")
        log.write(result.stdout)
        log.write(result.stderr)

    if result.returncode != 0:
        print("ERROR: Drift monitoring failed. Check logs/drift_monitor.log")
        sys.exit(result.returncode)

    with open(REPORTS / "drift_summary.json", encoding="utf-8") as file:
        summary = json.load(file)

    drift = summary["overall_drift_detected"]

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

    if drift:
        alert = (
            f"[{timestamp}] ALERT: Data drift detected in "
            f"{', '.join(summary['drifted_features'])}\n"
        )
        with open(LOGS / "drift_alerts.log", "a", encoding="utf-8") as file:
            file.write(alert)
        print(alert.strip())
        print("Retraining review recommended. No model was changed.")
    else:
        print("No data drift detected.")

    print("Monitoring completed successfully.")
    print(f"History: {history_path}")
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()
