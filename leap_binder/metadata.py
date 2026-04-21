import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata


def average_dist_nn(boxes: np.ndarray) -> float:
    if len(boxes) < 2:
        return 1.0
    data = boxes[:, :2]
    distance_matrix = np.full((len(data), len(data)), np.inf)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    return float(np.mean(np.min(distance_matrix[:, 1:], axis=0)))


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

    image_path = f"{data['root']}/{img_meta['file_name']}"
    image = cv2.imread(image_path)
    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    sharpness = laplacian.var()

    img_w, img_h = img_meta["width"], img_meta["height"]
    cat_ids = np.array([ann["category_id"] for ann in annotations], dtype=np.float32)
    bbox_cx = np.array([(ann["bbox"][0] + ann["bbox"][2] / 2) / img_w for ann in annotations], dtype=np.float32)
    bbox_cy = np.array([(ann["bbox"][1] + ann["bbox"][3] / 2) / img_h for ann in annotations], dtype=np.float32)
    bbox_areas = np.array([(ann["bbox"][2] * ann["bbox"][3]) / (img_w * img_h) for ann in annotations], dtype=np.float32)

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
