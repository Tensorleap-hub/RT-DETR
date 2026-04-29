import ntpath
import os
import posixpath
import yaml
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SUPPORTED_MODEL_OUTPUT_FORMATS = {
    "rtdetr_raw",
    "detections",
    "detections_concat_scores",
    "class_scores",
}

ROOT = Path(__file__).resolve().parent


def _is_absolute_path(path: str) -> bool:
    return ntpath.isabs(path) or posixpath.isabs(path)


def abs_path_from_root(path):
    path_str = os.fspath(path)
    expanded_path = os.path.expandvars(os.path.expanduser(path_str))
    if _is_absolute_path(expanded_path):
        return str(Path(expanded_path))
    return str(ROOT / expanded_path)


def _as_path_candidates(path_value: Any) -> List[str]:
    if path_value is None:
        return []
    if isinstance(path_value, (str, os.PathLike)):
        return [os.fspath(path_value)]
    if isinstance(path_value, list):
        return [os.fspath(candidate) for candidate in path_value]
    raise TypeError("dataset_path must be a string or a list of strings.")


def _split_paths_exist(dataset_root: str, split_value: Any) -> bool:
    if not split_value:
        return True
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    root_path = Path(abs_path_from_root(dataset_root))
    return all((root_path / split_path).exists() for split_path in split_paths)


def _required_dataset_splits(data_config: Dict[str, Any]) -> Iterable[Any]:
    return (
        data_config.get(split_name)
        for split_name in ("train", "val", "test")
        if data_config.get(split_name)
    )


def resolve_dataset_path(config: Dict[str, Any], data_config: Dict[str, Any]) -> str:
    dataset_path_candidates = _as_path_candidates(config.get("dataset_path"))
    if not dataset_path_candidates:
        return data_config.get("path", "")

    for candidate in dataset_path_candidates:
        if all(_split_paths_exist(candidate, split_value) for split_value in _required_dataset_splits(data_config)):
            return abs_path_from_root(candidate)

    attempted_paths = [abs_path_from_root(candidate) for candidate in dataset_path_candidates]
    raise FileNotFoundError(
        "Dataset not found in any configured dataset_path. "
        f"Attempted roots: {attempted_paths}"
    )

def resolve_coco_paths(config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    annotation_files = config.get("annotation_file", {})
    if isinstance(annotation_files, str):
        annotation_files = {"val": annotation_files}
    candidates = _as_path_candidates(config.get("dataset_path"))
    for candidate in candidates:
        root = abs_path_from_root(candidate)
        found = {
            split: os.path.join(root, fname)
            for split, fname in annotation_files.items()
            if os.path.exists(os.path.join(root, fname))
        }
        if found:
            roots = {split: os.path.dirname(path) for split, path in found.items()}
            return roots, found
    attempted = [abs_path_from_root(c) for c in candidates]
    raise FileNotFoundError(f"No COCO annotation files found in any dataset_path candidate. Tried: {attempted}")


def load_yaml(path) -> Dict[str, Any]:
    file_path = abs_path_from_root(path)
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_project_config() -> Dict[str, Any]:
    config = load_yaml("leap_config.yaml")
    model_output_format = config.setdefault("model_output_format", "rtdetr_raw")
    if model_output_format not in SUPPORTED_MODEL_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported model_output_format: {model_output_format}. "
            f"Expected one of {sorted(SUPPORTED_MODEL_OUTPUT_FORMATS)}."
        )
    return config


def load_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_yaml_path = config["data_yaml_path"]
    data_config = load_yaml(data_yaml_path)
    dataset_path = resolve_dataset_path(config, data_config)
    if dataset_path:
        data_config["path"] = dataset_path
    return data_config


CONFIG = load_project_config()
if "data_yaml_path" in CONFIG:
    data_config_path = abs_path_from_root(CONFIG["data_yaml_path"])
    if os.path.exists(data_config_path):
        DATA_CONFIG = load_dataset_config(CONFIG)
    else:
        DATA_CONFIG = {}
else:
    DATA_CONFIG = {}
