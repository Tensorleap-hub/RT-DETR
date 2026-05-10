from typing import Dict

import numpy as np
import torch

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss
from utils.general import xywh2xyxy

from leap_utils import compute_iou, compute_precision_recall_f1_fp_tp_fn

from .common import CONFIG, format_predictions, image_scale_wh, prediction_rows


def _batched_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim == 2:
        return targets[None, ...]
    return targets


def compute_detection_losses(targets: np.ndarray, *, y_preds: np.ndarray) -> Dict[str, np.ndarray]:
    preds = prediction_rows(y_preds)

    iou_losses = []
    f1_losses = []

    for pred, gt in zip(preds, _batched_targets(targets)):
        mask = ~(gt == -1).any(axis=1)
        gt = gt[mask]
        gt = torch.from_numpy(gt)

        if gt.shape[0] == 0 and pred.shape[0] == 0:
            iou_losses.append(0.0)
            f1_losses.append(0.0)
            continue

        if pred.shape[0] == 0 or gt.shape[0] == 0:
            iou_losses.append(1.0)
            f1_losses.append(1.0)
            continue

        pred_fmt = CONFIG.get("pred_bbox_format", "xyxy_abs")
        if pred_fmt == "cxcywh_norm":
            pred_boxes = xywh2xyxy(pred[:, :4])
        else:
            pred_boxes = pred[:, :4] / image_scale_wh(CONFIG["image_size"])
        gt_boxes = xywh2xyxy(gt[:, 1:])
        _, _, f1, _, _, _ = compute_precision_recall_f1_fp_tp_fn(gt_boxes, pred_boxes, iou_threshold=0.1)
        iou = compute_iou(gt_boxes, pred_boxes)

        iou_losses.append(float(1.0 - iou))
        f1_losses.append(float(1.0 - f1))

    return {
        "iou_loss": np.asarray(iou_losses, dtype=np.float32),
        "f1_loss": np.asarray(f1_losses, dtype=np.float32),
    }


@tensorleap_custom_loss("detection_iou_loss")
def detection_iou_loss(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_predictions(boxes_xyxy, scores_per_class))
    return losses["iou_loss"]


@tensorleap_custom_loss("detection_f1_loss")
def detection_f1_loss(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_predictions(boxes_xyxy, scores_per_class))
    return losses["f1_loss"]
