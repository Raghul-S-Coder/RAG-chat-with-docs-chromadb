import yaml
import os

config_path = "properties/vector_config.yaml"

def load_config():
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)