
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "train.csv"
OUTPUT = ROOT / "data" / "train_clean.csv"

def clean_data(df):
    df = df.drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    valid_stops = df.loc[df["stops"] > 0, "stops"]
    median_stops = valid_stops.median()
    df.loc[df["stops"] <= 0, "stops"] = median_stops

    df = df.drop(columns=["delay_logged_by_conductor"])
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    print("Original rows:", len(df))

    cleaned = clean_data(df)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT, index=False)

    print("Cleaned rows:", len(cleaned))
    print("Invalid stops:", (cleaned["stops"] <= 0).sum())
    print("Output saved:", OUTPUT)
