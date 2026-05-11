import os

import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from .common import CONFIG, parse_gt_bbox


def _safe_stat(values: np.ndarray, reducer) -> float:
    if len(values) == 0:
        return float(np.nan)
    return float(reducer(values))


@tensorleap_metadata("metadata")
def sample_metadata(idx: int, preprocessing: PreprocessResponse) -> dict:
    data = preprocessing.data
    img_meta = data["images"][idx]
    annotations = data["anns"].get(img_meta["id"], [])
    categories = data.get("categories", {})

    image_path = os.path.join(data["root"], img_meta["file_name"])
    image = cv2.imread(image_path)
    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    sharpness = laplacian.var()

    img_w, img_h = img_meta["width"], img_meta["height"]
    gt_fmt = CONFIG.get("gt_bbox_format", "xywh_abs")
    cat_ids = np.array([ann["category_id"] for ann in annotations], dtype=np.float32)
    parsed = [parse_gt_bbox(ann["bbox"], img_w, img_h, gt_fmt) for ann in annotations]
    bbox_cx = np.array([p[0] for p in parsed], dtype=np.float32)
    bbox_cy = np.array([p[1] for p in parsed], dtype=np.float32)
    bbox_areas = np.array([p[2] * p[3] for p in parsed], dtype=np.float32)

    unique_classes, class_counts = np.unique(cat_ids, return_counts=True)
    class_count_map = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, class_counts)}
    per_label_counts = {
        f"# of {name}": float(class_count_map.get(cat_id, 0))
        for cat_id, name in categories.items()
    }

    return {
        "image_sharpness": float(sharpness),
        "# of objects": len(annotations),
        "# of unique classes": int(len(unique_classes)),
        "bbox area mean": _safe_stat(bbox_areas, np.mean),
        "bbox area median": _safe_stat(bbox_areas, np.median),
        "bbox area min": _safe_stat(bbox_areas, np.min),
        "bbox area max": _safe_stat(bbox_areas, np.max),
        "bbox area var": _safe_stat(bbox_areas, np.var),
        "bbox cx mean": _safe_stat(bbox_cx, np.mean),
        "bbox cx median": _safe_stat(bbox_cx, np.median),
        "bbox cx min": _safe_stat(bbox_cx, np.min),
        "bbox cx max": _safe_stat(bbox_cx, np.max),
        "bbox cx var": _safe_stat(bbox_cx, np.var),
        "bbox cy mean": _safe_stat(bbox_cy, np.mean),
        "bbox cy median": _safe_stat(bbox_cy, np.median),
        "bbox cy min": _safe_stat(bbox_cy, np.min),
        "bbox cy max": _safe_stat(bbox_cy, np.max),
        "bbox cy var": _safe_stat(bbox_cy, np.var),
        "bbox center var": _safe_stat(bbox_cy, np.var) + _safe_stat(bbox_cx, np.var),
        **per_label_counts,
    }
