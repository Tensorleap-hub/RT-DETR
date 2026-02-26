import os
import yaml
from typing import Any, Dict

def abs_path_from_root(path):
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, path)
    return file_path

def load_yaml(path) -> Dict[str, Any]:
    file_path = abs_path_from_root(path)
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_project_config() -> Dict[str, Any]:
    config = load_yaml("leap_config.yaml")
    return config


CONFIG = load_project_config()
data_config_path = abs_path_from_root(CONFIG["data_yaml_path"])
if os.path.exists(data_config_path):
    DATA_CONFIG = load_yaml(CONFIG["data_yaml_path"])
else:
    DATA_CONFIG = {}
