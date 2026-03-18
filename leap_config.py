import os
import yaml
from typing import Any, Dict

DATASET_PROFILES = {
    "coco": {
        "data_yaml_path": "data/coco.yaml",
        "dataset_autodownload": False,
    },
    "coco128": {
        "data_yaml_path": "data/coco128.yaml",
        "dataset_autodownload": False,
    },
    "visdrone128": {
        "data_yaml_path": "data/visdrone128.yaml",
        "dataset_autodownload": False,
    },
    "visdrone": {
        "data_yaml_path": "data/visdrone.yaml",
        "dataset_autodownload": True,
    },
}


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
    dataset_name = config.get("dataset_name")
    if dataset_name:
        if dataset_name not in DATASET_PROFILES:
            raise ValueError(
                f"Unsupported dataset_name: {dataset_name}. "
                f"Expected one of {sorted(DATASET_PROFILES)}."
            )
        dataset_profile = DATASET_PROFILES[dataset_name]
        config.setdefault("data_yaml_path", dataset_profile["data_yaml_path"])
        config.setdefault("dataset_autodownload", dataset_profile["dataset_autodownload"])
    return config


def load_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_yaml_path = config["data_yaml_path"]
    data_config = load_yaml(data_yaml_path)
    dataset_path = config.get("dataset_path")
    if dataset_path:
        data_config["path"] = dataset_path
    return data_config


CONFIG = load_project_config()
data_config_path = abs_path_from_root(CONFIG["data_yaml_path"])
if os.path.exists(data_config_path):
    DATA_CONFIG = load_dataset_config(CONFIG)
else:
    DATA_CONFIG = {}
