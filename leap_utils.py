import numpy as np
import torch
from ultralytics.utils.metrics import box_iou


def compute_iou(gt_bbox, preds_bbox):
    iou_mat = box_iou(gt_bbox, preds_bbox)
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return np.zeros((1, 1))
    max_iou = iou_mat.max(dim=0, keepdim=True).values
    filtered_iou = iou_mat * iou_mat.eq(max_iou)
    return filtered_iou.max(dim=1).values.numpy().mean()


def compute_accuracy(gt_bbox, gt_labels, preds_bbox, preds_labels):
    iou_mat = box_iou(gt_bbox, preds_bbox)
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return np.zeros((1, 1))
    max_iou = iou_mat.max(dim=0, keepdim=True).values
    filtered_iou = iou_mat * iou_mat.eq(max_iou)
    succ = (preds_labels[filtered_iou.max(dim=1)[1].numpy()] == gt_labels).numpy()
    return succ.mean()


def compute_precision_recall_f1_fp_tp_fn(gt_boxes, pred_boxes, iou_threshold=0.5):
    iou_mat = box_iou(gt_boxes, pred_boxes)

    matched_gt = set()
    matched_pred = set()
    TP = 0

    for pred_idx in range(iou_mat.shape[1]):
        gt_idx = iou_mat[:, pred_idx].argmax().item()
        max_iou = iou_mat[gt_idx, pred_idx].item()

        if max_iou >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            TP += 1

    FP = pred_boxes.shape[0] - TP
    FN = gt_boxes.shape[0] - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, FP, TP, FN
