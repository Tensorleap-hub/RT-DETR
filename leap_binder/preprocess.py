import json
import os
from typing import Dict, List

import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_preprocess,
)
from leap_config import resolve_coco_paths

from .aws_utils import download_file_if_missing
from .common import CONFIG


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
    return {"images": images, "anns": anns, "root": dataset_root, "categories": categories}


_SPLIT_TO_STATE = {
    "train": DataStateType.training,
    "val": DataStateType.validation,
    "test": DataStateType.test,
}


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    split_roots, annotation_paths = resolve_coco_paths(CONFIG)
    responses = []
    for split in ["train", "val", "test"]:
        if split not in annotation_paths:
            continue
        data = _load_coco(annotation_paths[split], split_roots[split])
        responses.append(PreprocessResponse(data=data, length=len(data["images"]), state=_SPLIT_TO_STATE[split]))
    if not responses:
        raise ValueError("No COCO annotation files found for any split")
    return responses


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    data = preprocess.data
    img_meta = data["images"][idx]
    image_path = os.path.join(data["root"], img_meta["file_name"])

    s3_config = CONFIG.get("s3", {})
    if s3_config.get("enabled"):
        s3_key = f"{s3_config['prefix']}/{img_meta['file_name']}"
        download_file_if_missing(s3_config["bucket_name"], s3_key, image_path)

    image_size = CONFIG["image_size"]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


@tensorleap_input_encoder("orig_size", channel_dim=1)
def input_size_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    img_meta = preprocess.data["images"][idx]
    return np.array([img_meta["height"], img_meta["width"]], dtype=np.float32)


def _padded_gt_for_sample(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    data = preprocessing.data
    img_meta = data["images"][idx]
    annotations = data["anns"].get(img_meta["id"], [])
    max_num_of_objs = int(CONFIG["max_num_of_objects"])
    img_w = img_meta["width"]
    img_h = img_meta["height"]

    rows = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
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
