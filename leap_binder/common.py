from typing import List

import numpy as np
import torch

from leap_config import CONFIG, abs_path_from_root
from utils.general import non_max_suppression


def image_scale_wh(image_size) -> np.ndarray:
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        h, w = image_size
    else:
        h = w = int(image_size)
    return np.array([w, h, w, h], dtype=np.float32)


def split_boxes_and_scores_from_concat(boxes_with_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes_with_scores = np.asarray(boxes_with_scores)
    if boxes_with_scores.shape[-1] < 5:
        raise ValueError(
            f"Expected boxes tensor with last dimension >= 5 for concat-score format, got shape {boxes_with_scores.shape}"
        )
    return boxes_with_scores[..., :4], boxes_with_scores[..., 4]


def format_rtdetr_predictions(
    labels: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    *,
    _score_threshold: float = None,
) -> np.ndarray:
    labels = np.asarray(labels).squeeze()
    boxes_xyxy = np.asarray(boxes_xyxy).squeeze()
    scores = np.asarray(scores).squeeze()

    if labels.ndim == 0:
        labels = np.array([labels], dtype=np.float32)
    if scores.ndim == 0:
        scores = np.array([scores], dtype=np.float32)
    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)

    score_threshold = _score_threshold if _score_threshold is not None else float(CONFIG.get("score_threshold", 0.3))
    max_detections = int(CONFIG.get("max_detections", 300))
    keep = scores >= score_threshold
    labels = labels[keep]
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    if scores.size == 0:
        return np.zeros((1, 0, 6), dtype=np.float32)

    order = np.argsort(-scores)[:max_detections]
    labels = labels[order]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    pred = np.concatenate([boxes_xyxy, scores[:, None], labels[:, None]], axis=1).astype(np.float32)
    return pred[None, ...]


def format_rtdetr_concat_predictions(labels: np.ndarray, boxes_with_scores: np.ndarray) -> np.ndarray:
    boxes_xyxy, scores = split_boxes_and_scores_from_concat(boxes_with_scores)
    return format_rtdetr_predictions(labels, boxes_xyxy, scores)


def _bbox_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    return np.column_stack((
        boxes[:, 0] - half_w,
        boxes[:, 1] - half_h,
        boxes[:, 0] + half_w,
        boxes[:, 1] + half_h,
    ))


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

    if CONFIG.get("boxes_in_cxcywh_format", False):
        boxes = _bbox_cxcywh_to_xyxy(boxes)

    scalar_scores = scores_per_class.max(axis=-1)
    labels = scores_per_class.argmax(axis=-1).astype(np.float32)

    return format_rtdetr_predictions(labels, boxes, scalar_scores,
                                     _score_threshold=score_threshold)


def prediction_rows(y_preds: np.ndarray) -> List[torch.Tensor]:
    y_preds = np.asarray(y_preds)
    if y_preds.ndim == 3 and y_preds.shape[-1] == 6:
        return [torch.from_numpy(y_preds[0].astype(np.float32))]
    return non_max_suppression(torch.from_numpy(y_preds))


def label_names() -> List[str]:
    return CONFIG.get("_label_names", [])


COCO_CATEGORY_TO_LABEL = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
    17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40,
    47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53,
    60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66,
    77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
