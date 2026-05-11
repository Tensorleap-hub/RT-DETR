from typing import List

import numpy as np
import torch

from leap_config import CONFIG
from utils.general import non_max_suppression, xywh2xyxy, xyxy2xywh


def image_scale_wh(image_size) -> np.ndarray:
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        h, w = image_size
    else:
        h = w = int(image_size)
    return np.array([w, h, w, h], dtype=np.float32)


def _apply_score_threshold(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    max_detections: int,
) -> np.ndarray:
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if scores.size == 0:
        return np.zeros((1, 0, 6), dtype=np.float32)

    order = np.argsort(-scores)[:max_detections]
    pred = np.concatenate(
        [boxes[order], scores[order, None], labels[order, None]], axis=1
    ).astype(np.float32)
    return pred[None, ...]


def format_predictions(
    boxes: np.ndarray,
    scores_per_class: np.ndarray,
    score_threshold: float = None,
) -> np.ndarray:
    boxes = np.asarray(boxes).squeeze()
    scores_per_class = np.asarray(scores_per_class).squeeze()

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if scores_per_class.ndim == 1:
        scores_per_class = scores_per_class.reshape(1, -1)

    scalar_scores = scores_per_class.max(axis=-1)
    labels = scores_per_class.argmax(axis=-1).astype(np.float32)

    threshold = score_threshold if score_threshold is not None else float(CONFIG.get("score_threshold", 0.3))
    max_detections = int(CONFIG.get("max_detections", 300))
    return _apply_score_threshold(boxes, labels, scalar_scores, threshold, max_detections)


def prediction_rows(y_preds: np.ndarray) -> List[torch.Tensor]:
    y_preds = np.asarray(y_preds)
    if y_preds.ndim == 3 and y_preds.shape[-1] == 6:
        return [torch.from_numpy(y_preds[0].astype(np.float32))]
    return non_max_suppression(torch.from_numpy(y_preds))


def parse_gt_bbox(b, img_w: int, img_h: int, fmt: str):
    sx = float(img_w) if "_norm" in fmt else 1.0
    sy = float(img_h) if "_norm" in fmt else 1.0
    if "xyxy" in fmt:
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
    elif "cxcywh" in fmt:
        cx_a, cy_a, w_a, h_a = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        x1, y1, x2, y2 = cx_a - w_a / 2, cy_a - h_a / 2, cx_a + w_a / 2, cy_a + h_a / 2
    else:  # xywh
        x1, y1, w_a, h_a = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        x2, y2 = x1 + w_a, y1 + h_a
    return (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h


def pred_boxes_to_norm_cxcywh(boxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    fmt = CONFIG.get("pred_bbox_format", "xyxy_abs")
    scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    if fmt == "xyxy_abs":
        return xyxy2xywh(boxes) / scale
    elif fmt == "xyxy_norm":
        return xyxy2xywh(boxes)
    elif fmt == "cxcywh_abs":
        return boxes / scale
    else:  # cxcywh_norm
        return boxes


def pred_boxes_to_norm_xyxy(boxes: np.ndarray, image_size) -> np.ndarray:
    fmt = CONFIG.get("pred_bbox_format", "xyxy_abs")
    scale = image_scale_wh(image_size)
    if fmt == "xyxy_abs":
        return boxes / scale
    elif fmt == "xyxy_norm":
        return boxes
    elif fmt == "cxcywh_abs":
        return xywh2xyxy(boxes / scale)
    else:  # cxcywh_norm
        return xywh2xyxy(boxes)


def label_names() -> List[str]:
    return CONFIG.get("_label_names", [])
