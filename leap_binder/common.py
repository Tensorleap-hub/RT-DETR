from typing import List

import numpy as np
import torch

from leap_config import CONFIG
from utils.general import non_max_suppression


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


def label_names() -> List[str]:
    return CONFIG.get("_label_names", [])
