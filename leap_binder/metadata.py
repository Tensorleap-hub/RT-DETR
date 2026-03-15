import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from .common import DATA_CONFIG


def average_dist_nn(boxes: np.ndarray) -> float:
    if len(boxes) < 2:
        return 1.0
    data = boxes[:, :2]
    distance_matrix = np.full((len(data), len(data)), np.inf)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
    return float(np.mean(np.min(distance_matrix[:, 1:], axis=0)))


@tensorleap_metadata("metadata")
def sample_metadata(idx: int, preprocessing: PreprocessResponse) -> dict:
    sample = preprocessing.data[idx]
    image = (sample[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    gt = sample[1].numpy()

    if gt.shape[0] != 0:
        gt_class = gt[:, 1]
        gt_bbox = gt[:, 2:]
        bbox_areas = gt_bbox[:, 2] * gt_bbox[:, 3]
        bbox_cx = gt_bbox[:, 0]
        bbox_cy = gt_bbox[: 1]
    else:
        gt_class, bbox_areas, bbox_cx, bbox_cy = np.array([]), np.array([]), np.array([]), np.array([])

    unique_classes, class_counts = np.unique(gt_class, return_counts=True)
    labels = DATA_CONFIG.get("pred_names", DATA_CONFIG.get("names", []))
    class_count_map = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, class_counts)}
    per_label_counts = {
        f"# of {label}": float(class_count_map[label_idx]) if label_idx in class_count_map else float(np.nan)
        for label_idx, label in enumerate(labels)
    }

    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    sharpness = laplacian.var()
    metadata_dict = {
        "image_sharpness": float(sharpness),
        "# of objects": gt.shape[0],
        "# of unique objects": len(unique_classes),
        "bbox area mean": float(bbox_areas.mean()),
        "bbox area median": float(np.median(bbox_areas)),
        "bbox area min": float(bbox_areas.min() if len(bbox_areas) > 0 else np.nan),
        "bbox area max": float(bbox_areas.max() if len(bbox_areas) > 0 else np.nan),
        "bbox area var": float(bbox_areas.var()),
        "bbox cx mean": float(bbox_cx.mean()),
        "bbox cx median": float(np.median(bbox_cx)),
        "bbox cx min": float(bbox_cx.min() if len(bbox_cx) > 0 else np.nan),
        "bbox cx max": float(bbox_cx.max() if len(bbox_cx) > 0 else np.nan),
        "bbox cx var": float(bbox_cx.var()),
        "bbox cy mean": float(bbox_cy.mean()),
        "bbox cy median": float(np.median(bbox_cy)),
        "bbox cy min": float(bbox_cy.min() if len(bbox_cy) > 0 else np.nan),
        "bbox cy max": float(bbox_cy.max() if len(bbox_cy) > 0 else np.nan),
        "bbox cy var": float(bbox_cy.var()),
        "bbox center var": float(bbox_cy.var()) + float(bbox_cx.var()),
        **per_label_counts,
    }
    return metadata_dict
