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
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    result = subprocess.run(
        [sys.executable, "-m", "src.psi_monitor"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    with open(LOGS / "drift_monitor.log", "a", encoding="utf-8") as file:
        file.write(f"\n[{timestamp}]\n")
        file.write(result.stdout)
        file.write(result.stderr)

    if result.returncode != 0:
        print("PSI monitoring failed. Check logs.")
        sys.exit(result.returncode)

    with open(REPORTS / "psi_report.json", encoding="utf-8") as file:
        report = json.load(file)

    drift = report["drift_detected"]
    threshold = report["threshold"]

    features = [
        name for name, value in report["psi"].items()
        if value > threshold
    ]

    history = REPORTS / "monitoring_history.csv"
    new_file = not history.exists() or history.stat().st_size == 0

    with open(history, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow([
                "timestamp",
                "drift_detected",
                "drifted_features"
            ])

        writer.writerow([
            timestamp,
            drift,
            ", ".join(features)
        ])

    if drift:
        alert = f"ALERT: PSI drift detected in {', '.join(features)}"

        with open(LOGS / "drift_alerts.log", "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {alert}\n")

        print(alert)
        print("Candidate model evaluation required.")
        print("Production champion remains unchanged.")
    else:
        print("No significant drift detected.")

    print("PSI monitoring completed successfully.")


if __name__ == "__main__":
    main()
