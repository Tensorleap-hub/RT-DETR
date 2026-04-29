import json
import ntpath
import os
import posixpath
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _load_yaml(path) -> Dict[str, Any]:
    file_path = abs_path_from_root(path)
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def _label_names_from_coco(annotation_path: str) -> List[str]:
    with open(annotation_path) as f:
        data = json.load(f)
    categories = data.get("categories", [])
    if not categories:
        return []
    max_id = max(cat["id"] for cat in categories)
    names = [""] * (max_id + 1)
    for cat in categories:
        names[cat["id"]] = cat["name"]
    return names


def load_label_names(config: Dict[str, Any]) -> List[str]:
    data_yaml_path = config.get("data_yaml_path")
    if data_yaml_path:
        yaml_path = abs_path_from_root(data_yaml_path)
        if os.path.exists(yaml_path):
            data = _load_yaml(data_yaml_path)
            names = data.get("pred_names", data.get("names", []))
            if names:
                return names

    candidates = _as_path_candidates(config.get("dataset_path"))
    annotation_files = config.get("annotation_file", {})
    if isinstance(annotation_files, str):
        annotation_files = {"val": annotation_files}
    for candidate in candidates:
        root = abs_path_from_root(candidate)
        for fname in annotation_files.values():
            ann_path = os.path.join(root, fname)
            if os.path.exists(ann_path):
                return _label_names_from_coco(ann_path)
    return []


def load_project_config() -> Dict[str, Any]:
    return _load_yaml("leap_config.yaml")


CONFIG = load_project_config()
CONFIG["_label_names"] = load_label_names(CONFIG)
