import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_preprocess,
)
from leap_config import _dataset_root, resolve_coco_paths

from .minio_utils import download_annotations, download_file_if_missing, object_exists
from .common import CONFIG, parse_gt_bbox


def _load_coco(annotation_path: str, dataset_root: str) -> Dict:
    with open(annotation_path) as f:
        coco = json.load(f)
    images = coco["images"]
    for img in images:
        img["file_name"] = img["file_name"].replace("\\", "/")
    anns: Dict[int, List] = {}
    for ann in coco.get("annotations", []):
        anns.setdefault(ann["image_id"], []).append(ann)
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    return {"images": images, "anns": anns, "root": Path(dataset_root) / "images", "categories": categories}


_SPLIT_TO_STATE = {
    "train": DataStateType.training,
    "val": DataStateType.validation,
    "test": DataStateType.test,
}


def _minio_key_for(image_path: str, minio_config: Dict) -> str:
    relative = Path(image_path).relative_to(_dataset_root(CONFIG))
    return f"{minio_config['prefix']}/{relative}"


def load_raw_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    data = preprocess.data
    img_meta = data["images"][idx]
    image_path = str(data["root"] / img_meta["file_name"])

    minio_config = CONFIG.get("minio", {})
    if minio_config.get("enabled"):
        download_file_if_missing(
            minio_config["bucket_name"], _minio_key_for(image_path, minio_config), image_path
        )

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def _image_available(image_path: str, minio_config: Dict) -> bool:
    if Path(image_path).exists():
        return True
    if minio_config.get("enabled"):
        return object_exists(minio_config["bucket_name"], _minio_key_for(image_path, minio_config))
    return False


def _filter_paired_samples(data: Dict, split: str) -> Dict:
    minio_config = CONFIG.get("minio", {})
    with_anns = [img for img in data["images"] if data["anns"].get(img["id"])]
    no_annotations = len(data["images"]) - len(with_anns)

    image_paths = [str(data["root"] / img["file_name"]) for img in with_anns]
    workers = int(minio_config.get("head_check_workers", 32))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        available = list(pool.map(lambda path: _image_available(path, minio_config), image_paths))
    kept = [img for img, ok in zip(with_anns, available) if ok]
    no_image = len(with_anns) - len(kept)

    if no_annotations or no_image:
        print(
            f"[{split}] skipping {no_image} sample(s) with a missing image file and "
            f"{no_annotations} without annotations; {len(kept)}/{len(data['images'])} remain"
        )
    data["images"] = kept
    return data


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    minio_config = CONFIG.get("minio", {})
    if minio_config.get("enabled"):
        download_annotations(
            minio_config["bucket_name"],
            minio_config["prefix"],
            CONFIG.get("annotation_file", {}),
            str(_dataset_root(CONFIG)),
        )
    split_roots, annotation_paths = resolve_coco_paths(CONFIG)
    responses = []
    for split in ["train", "val", "test"]:
        if split not in annotation_paths:
            continue
        data = _filter_paired_samples(_load_coco(annotation_paths[split], split_roots[split]), split)
        if not data["images"]:
            print(f"[{split}] no samples with both an image and annotations, skipping split")
            continue
        responses.append(PreprocessResponse(data=data, length=len(data["images"]), state=_SPLIT_TO_STATE[split]))
    if not responses:
        raise ValueError("No COCO annotation files found for any split")
    return responses


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image_size = CONFIG["image_size"]
    img = load_raw_image(idx, preprocess)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


def _padded_gt_for_sample(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    data = preprocessing.data
    img_meta = data["images"][idx]
    annotations = data["anns"].get(img_meta["id"], [])
    max_num_of_objs = int(CONFIG["max_num_of_objects"])
    img_w = img_meta["width"]
    img_h = img_meta["height"]

    gt_fmt = CONFIG.get("gt_bbox_format", "xywh_abs")
    rows = []
    for ann in annotations:
        cx, cy, nw, nh = parse_gt_bbox(ann["bbox"], img_w, img_h, gt_fmt)
        rows.append([float(ann["category_id"]), cx, cy, nw, nh])

    if not rows:
        return np.full((max_num_of_objs, 5), -1, dtype=np.float32)

    gt = np.array(rows, dtype=np.float32)
    if gt.shape[0] < max_num_of_objs:
        pad = np.full((max_num_of_objs - gt.shape[0], 5), -1, dtype=np.float32)
        gt = np.vstack([gt, pad])
    else:
        gt = gt[:max_num_of_objs]
    return gt


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return _padded_gt_for_sample(idx, preprocessing)


@tensorleap_gt_encoder("gt_boxes")
def gt_boxes_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    boxes = gt[:, 1:5].copy()
    boxes[gt[:, 0] < 0] = 0.0
    return boxes


@tensorleap_gt_encoder("gt_labels")
def gt_labels_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return _padded_gt_for_sample(idx, preprocessing)[:, 0]


@tensorleap_gt_encoder("gt_valid_mask")
def gt_valid_mask_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    return (gt[:, 0] >= 0).astype(np.float32)
