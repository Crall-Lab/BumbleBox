# config_loader.py
import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Dynamically add BumbleBox directory based on current working directory
    config["bumblebox_dir"] = os.getcwd()

    return config
