import yaml
import os

from pathlib import Path

config_path = Path(__file__).resolve().parent / 'vector-config.yaml'


def load_config():
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)