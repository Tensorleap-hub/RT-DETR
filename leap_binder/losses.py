import textwrap
from typing import Dict, List

import numpy as np
import torch

from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_metric,
)
from rtdetr_native.criterion import RTDETRCriterionv2
from rtdetr_native.matcher import HungarianMatcher
from utils.general import xywh2xyxy

from leap_utils import compute_iou, compute_precision_recall_f1_fp_tp_fn

from .common import (
    COCO_CATEGORY_TO_LABEL,
    CONFIG,
    format_rtdetr_concat_predictions,
    format_rtdetr_predictions,
    prediction_rows,
)


def _loss_cfg() -> Dict:
    loss_cfg = CONFIG.get("loss", {})
    matcher_cfg = loss_cfg.get("matcher", {})
    weight_cfg = loss_cfg.get("weight_dict", {})
    return {
        "map_coco_category_to_label": bool(loss_cfg.get("map_coco_category_to_label", False)),
        "alpha": float(loss_cfg.get("alpha", 0.75)),
        "gamma": float(loss_cfg.get("gamma", 2.0)),
        "matcher": {
            "cost_class": float(matcher_cfg.get("cost_class", 2.0)),
            "cost_bbox": float(matcher_cfg.get("cost_bbox", 5.0)),
            "cost_giou": float(matcher_cfg.get("cost_giou", 2.0)),
            "alpha": float(matcher_cfg.get("alpha", 0.25)),
            "gamma": float(matcher_cfg.get("gamma", 2.0)),
        },
        "weight_dict": {
            "loss_vfl": float(weight_cfg.get("loss_vfl", 1.0)),
            "loss_bbox": float(weight_cfg.get("loss_bbox", 5.0)),
            "loss_giou": float(weight_cfg.get("loss_giou", 2.0)),
        },
    }


def _extract_targets_for_native_loss(
    gt_boxes: np.ndarray, gt_labels: np.ndarray, gt_valid_mask: np.ndarray
) -> List[Dict[str, torch.Tensor]]:
    boxes = gt_boxes[0] if gt_boxes.ndim == 3 else gt_boxes
    labels = gt_labels[0] if gt_labels.ndim == 2 else gt_labels
    valid = gt_valid_mask[0] if gt_valid_mask.ndim == 2 else gt_valid_mask

    keep = valid > 0.5
    boxes = boxes[keep].astype(np.float32)
    labels = labels[keep].astype(np.int64)

    cfg = _loss_cfg()
    if cfg["map_coco_category_to_label"]:
        mapped_labels = []
        mapped_boxes = []
        for box, cls in zip(boxes, labels):
            mapped = COCO_CATEGORY_TO_LABEL.get(int(cls), -1)
            if mapped >= 0:
                mapped_labels.append(mapped)
                mapped_boxes.append(box)
        if mapped_boxes:
            boxes = np.asarray(mapped_boxes, dtype=np.float32)
            labels = np.asarray(mapped_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

    target = {
        "boxes": torch.from_numpy(boxes),
        "labels": torch.from_numpy(labels),
    }
    return [target]


def compute_rtdetr_native_losses(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, float]:
    logits = pred_logits if pred_logits.ndim == 3 else np.expand_dims(pred_logits, axis=0)
    boxes = pred_boxes if pred_boxes.ndim == 3 else np.expand_dims(pred_boxes, axis=0)

    outputs = {
        "pred_logits": torch.from_numpy(logits.astype(np.float32)),
        "pred_boxes": torch.from_numpy(boxes.astype(np.float32)),
    }
    targets = _extract_targets_for_native_loss(gt_boxes, gt_labels, gt_valid_mask)
    cfg = _loss_cfg()
    matcher = HungarianMatcher(
        weight_dict={
            "cost_class": cfg["matcher"]["cost_class"],
            "cost_bbox": cfg["matcher"]["cost_bbox"],
            "cost_giou": cfg["matcher"]["cost_giou"],
        },
        use_focal_loss=True,
        alpha=cfg["matcher"]["alpha"],
        gamma=cfg["matcher"]["gamma"],
    )
    criterion = RTDETRCriterionv2(
        matcher=matcher,
        weight_dict=cfg["weight_dict"],
        losses=["vfl", "boxes"],
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        num_classes=int(outputs["pred_logits"].shape[-1]),
    )
    loss_tensors = criterion(outputs, targets)
    scalar_losses = {
        k: float(v.detach().cpu().item())
        for k, v in loss_tensors.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_losses["total"] = float(sum(scalar_losses.values()))
    return scalar_losses


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

        pred_boxes = pred[:, :4] / CONFIG["image_size"]
        gt_boxes = xywh2xyxy(gt[:, 1:])
        _, _, f1, _, _, _ = compute_precision_recall_f1_fp_tp_fn(gt_boxes, pred_boxes, iou_threshold=0.1)
        iou = compute_iou(gt_boxes, pred_boxes)

        iou_losses.append(float(1.0 - iou))
        f1_losses.append(float(1.0 - f1))

    return {
        "iou_loss": np.asarray(iou_losses, dtype=np.float32),
        "f1_loss": np.asarray(f1_losses, dtype=np.float32),
    }


@tensorleap_custom_loss("rtdetr_total_loss_native")
def rtdetr_total_loss_native(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> np.ndarray:
    losses = compute_rtdetr_native_losses(
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_valid_mask=gt_valid_mask,
    )
    return np.array([losses["total"]], dtype=np.float32)


@tensorleap_custom_loss("detection_iou_loss")
def detection_iou_loss(
    labels: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_rtdetr_predictions(labels, boxes_xyxy, scores))
    return losses["iou_loss"]


@tensorleap_custom_loss("detection_f1_loss")
def detection_f1_loss(
    labels: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_rtdetr_predictions(labels, boxes_xyxy, scores))
    return losses["f1_loss"]


@tensorleap_custom_loss("detection_iou_loss_concat_scores")
def detection_iou_loss_concat_scores(
    labels: np.ndarray,
    boxes_with_scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_rtdetr_concat_predictions(labels, boxes_with_scores))
    return losses["iou_loss"]


@tensorleap_custom_loss("detection_f1_loss_concat_scores")
def detection_f1_loss_concat_scores(
    labels: np.ndarray,
    boxes_with_scores: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(targets, y_preds=format_rtdetr_concat_predictions(labels, boxes_with_scores))
    return losses["f1_loss"]


@tensorleap_custom_metric(
    "rtdetr_loss_components_native",
    direction={
        "loss_vfl": MetricDirection.Downward,
        "loss_bbox": MetricDirection.Downward,
        "loss_giou": MetricDirection.Downward,
        "total": MetricDirection.Downward,
    },
)
def rtdetr_loss_components_native(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    losses = compute_rtdetr_native_losses(
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_valid_mask=gt_valid_mask,
    )
    return {k: np.array([v], dtype=np.float32) for k, v in losses.items()}


def yolov5_loss_factory(num_scales: int):
    preds_list = ", ".join([f"pred{i}" for i in range(num_scales)])
    all_args = f"{preds_list}, gt, demo_pred"

    fn_code = f'''
    @tensorleap_custom_loss("yolov5_loss")
    def yolov5_loss({all_args}):
        preds = [torch.from_numpy(p) for p in [{preds_list}]]
        gt = gt.squeeze(0)
        mask = ~(gt == -1).any(axis=1)
        gt = gt[mask]
        gt_torch = torch.from_numpy(gt)
        gt_torch = torch.cat([torch.zeros_like(gt_torch[:, 1]).unsqueeze(1), gt_torch], dim=1)
        loss = yolov5_loss_compute(preds, gt_torch)[0]
        return loss.unsqueeze(0).numpy()
    '''
    local_ns = {}
    exec(textwrap.dedent(fn_code), globals(), local_ns)
    return local_ns["yolov5_loss"]


@tensorleap_custom_loss("yolov5_new_loss")
def yolov5_new_loss(
    pred0: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    pred3: np.ndarray,
    gt: np.ndarray,
    demo_pred: np.ndarray,
):
    loss = np.zeros(pred1.shape[0])
    return loss
