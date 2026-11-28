from src.mp249.config import ROOT, load_config, get_data_hash

if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully")
    print("Dataset SHA256:", get_data_hash())
    print("Experiments:", config["training"]["experiments"])
