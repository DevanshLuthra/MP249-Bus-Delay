import pandas as pd
from src.config import ROOT, load_config

config = load_config()

raw = pd.read_csv(ROOT / config["data"]["train"])
holdout = pd.read_csv(ROOT / config["data"]["holdout"])

df = raw.drop_duplicates().copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "record_id"])

print("\n1. TARGET LEAKAGE")
print("Correlation:", df["delay_logged_by_conductor"].corr(df["delay_min"]))
print("Present in holdout:", "delay_logged_by_conductor" in holdout.columns)

print("\n2. DUPLICATE ROWS")
print("Original rows:", len(raw))
print("Unique rows:", len(df))
print("Duplicates:", raw.duplicated().sum())

print("\n3. SENTINEL VALUES")
print("Raw sentinel count:", (raw["stops"] == -1).sum())
print("Unique sentinel count:", (df["stops"] == -1).sum())

print("\n4. UNIT MIXUP")
numeric = config["features"]["numerical"]

for group in ["depot_zone", "weather"]:
    print("\nGrouped by:", group)
    print(df.groupby(group)[numeric].median().round(2).to_string())

print("\n5. STUCK SENSOR")

for column in numeric:
    values = df[column]
    runs = values.ne(values.shift()).cumsum()
    sizes = values.groupby(runs).size()
    longest = sizes.idxmax()
    rows = df.loc[runs == longest]

    print(
        column,
        "| Longest identical run:", len(rows),
        "| Value:", rows[column].iloc[0],
        "| Start:", rows["date"].min().date(),
        "| End:", rows["date"].max().date()
    )

print("\n6. TARGET TYPOS")
print(df["delay_min"].describe().to_string())
print("Negative targets:", (df["delay_min"] < 0).sum())
print("Largest target values:", df["delay_min"].nlargest(10).tolist())

print("\n7. SPELLING VARIANTS")

for column in config["features"]["categorical"]:
    print(column)
    print(df[column].value_counts(dropna=False).to_string())

print("\nAUDIT COMPLETED")
