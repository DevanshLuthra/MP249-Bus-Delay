from pathlib import Path
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config():
    with open(ROOT / "params.yaml", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)

def get_data_hash():
    config = load_config()
    path = ROOT / config["data"]["train"]
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully")
    print("Dataset SHA256:", get_data_hash())
    print("Experiments:", config["training"]["experiments"])
