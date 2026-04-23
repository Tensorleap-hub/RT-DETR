import numpy as np
import torch

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric
from leap_utils import compute_accuracy, compute_iou, compute_precision_recall_f1_fp_tp_fn
from utils.general import xywh2xyxy
from utils.metrics import box_iou

from .common import CONFIG, format_class_scores_predictions, format_rtdetr_concat_predictions, format_rtdetr_predictions, image_scale_wh, label_names, prediction_rows


def _batched_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim == 2:
        return targets[None, ...]
    return targets


def get_per_sample_metrics_from_predictions(y_preds: np.ndarray, targets: np.ndarray):
    def _update_metrics(metrics, precision, recall, f1, fp, tp, fn, iou, accuracy):
        metrics["precision"] = np.concatenate([metrics["precision"], np.array([precision], dtype=np.float32)])
        metrics["recall"] = np.concatenate([metrics["recall"], np.array([recall], dtype=np.float32)])
        metrics["f1"] = np.concatenate([metrics["f1"], np.array([f1], dtype=np.float32)])
        metrics["FP"] = np.concatenate([metrics["FP"], np.array([fp], dtype=np.int32)])
        metrics["TP"] = np.concatenate([metrics["TP"], np.array([tp], dtype=np.int32)])
        metrics["FN"] = np.concatenate([metrics["FN"], np.array([fn], dtype=np.int32)])
        metrics["iou"] = np.concatenate([metrics["iou"], np.array([iou], dtype=np.float32)])
        metrics["accuracy"] = np.concatenate([metrics["accuracy"], np.array([accuracy], dtype=np.float32)])

    metrics = {
        "precision": np.array([], dtype=np.float32),
        "recall": np.array([], dtype=np.float32),
        "f1": np.array([], dtype=np.float32),
        "FP": np.array([], dtype=np.int32),
        "TP": np.array([], dtype=np.int32),
        "FN": np.array([], dtype=np.int32),
        "iou": np.array([], dtype=np.float32),
        "accuracy": np.array([], dtype=np.float32),
    }
    preds = prediction_rows(y_preds)
    for pred, gt in zip(preds, _batched_targets(targets)):
        mask = ~(gt == -1).any(axis=1)
        gt = gt[mask]
        gt = torch.from_numpy(gt)

        if gt.shape[0] == 0 and pred.shape[0] == 0:
            _update_metrics(metrics, np.nan, np.nan, 0, 0, 0, 0, 1, 1)
            continue

        if pred.shape[0] == 0:
            _update_metrics(metrics, np.nan, 0, 0, 0, 0, gt.shape[0], 0, 0)
            continue

        if gt.shape[0] == 0:
            _update_metrics(metrics, 0, np.nan, 0, pred.shape[0], 0, 0, 0, 0)
            continue

        model_input_hw = CONFIG.get("_model_input_hw", CONFIG["image_size"])
        pred_boxes = pred[:, :4] / image_scale_wh(model_input_hw)
        pred_labels = pred[:, 5]

        gt_boxes = xywh2xyxy(gt[:, 1:])
        gt_labels = gt[:, 0]

        p, r, f1, fp, tp, fn = compute_precision_recall_f1_fp_tp_fn(gt_boxes, pred_boxes, iou_threshold=0.1)
        iou = compute_iou(gt_boxes, pred_boxes)
        acc = compute_accuracy(gt_boxes, gt_labels, pred_boxes, pred_labels)
        _update_metrics(metrics, float(p), float(r), float(f1), int(fp), int(tp), int(fn), float(iou), float(acc))
    return metrics


@tensorleap_custom_metric(
    name="per_sample_metrics",
    direction={
        "precision": MetricDirection.Upward,
        "recall": MetricDirection.Upward,
        "f1": MetricDirection.Upward,
        "FP": MetricDirection.Downward,
        "TP": MetricDirection.Upward,
        "FN": MetricDirection.Downward,
        "iou": MetricDirection.Upward,
        "accuracy": MetricDirection.Upward,
    },
)
def get_per_sample_metrics(labels, boxes_xyxy, scores, targets: np.ndarray):
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    return get_per_sample_metrics_from_predictions(y_preds, targets)


@tensorleap_custom_metric(
    name="per_sample_metrics_concat_scores",
    direction={
        "precision": MetricDirection.Upward,
        "recall": MetricDirection.Upward,
        "f1": MetricDirection.Upward,
        "FP": MetricDirection.Downward,
        "TP": MetricDirection.Upward,
        "FN": MetricDirection.Downward,
        "iou": MetricDirection.Upward,
        "accuracy": MetricDirection.Upward,
    },
)
def get_per_sample_metrics_concat_scores(labels, boxes_with_scores, targets: np.ndarray):
    y_preds = format_rtdetr_concat_predictions(labels, boxes_with_scores)
    return get_per_sample_metrics_from_predictions(y_preds, targets)


@tensorleap_custom_metric(
    name="per_sample_metrics_class_scores",
    direction={
        "precision": MetricDirection.Upward,
        "recall": MetricDirection.Upward,
        "f1": MetricDirection.Upward,
        "FP": MetricDirection.Downward,
        "TP": MetricDirection.Upward,
        "FN": MetricDirection.Downward,
        "iou": MetricDirection.Upward,
        "accuracy": MetricDirection.Upward,
    },
)
def get_per_sample_metrics_class_scores(boxes_xyxy: np.ndarray, scores_per_class: np.ndarray, targets: np.ndarray):
    y_preds = format_class_scores_predictions(boxes_xyxy, scores_per_class)
    return get_per_sample_metrics_from_predictions(y_preds, targets)


def confusion_matrix_metric_from_predictions(y_preds: np.ndarray, targets: np.ndarray):
    threshold = 0.1
    confusion_matrices = []
    names = label_names()
    preds = prediction_rows(y_preds)
    for pred, gt in zip(preds, _batched_targets(targets)):
        confusion_matrix_elements = []

        mask = ~(gt == -1).any(axis=1)
        gt = gt[mask]
        gt = torch.from_numpy(gt)
        gt_bbox = xywh2xyxy(gt[:, 1:])
        gt_labels = gt[:, 0]

        model_input_hw = CONFIG.get("_model_input_hw", CONFIG["image_size"])
        pred_boxes = pred[:, :4] / image_scale_wh(model_input_hw)

        if pred.shape[0] != 0 and gt_bbox.shape[0] != 0:
            ious = box_iou(gt_bbox, pred_boxes).numpy().T
            prediction_detected = np.any(ious > threshold, axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt_labels[max_iou_ind[i]])
                class_name = names[gt_idx] if 0 <= gt_idx < len(names) else "Unknown Class"
                confidence = pred[i, 4]
                if prediction:
                    confusion_matrix_elements.append(
                        ConfusionMatrixElement(str(class_name), ConfusionMatrixValue.Positive, float(confidence))
                    )
                else:
                    pred_idx = int(pred[i, 5])
                    class_name = names[pred_idx] if 0 <= pred_idx < len(names) else "Unknown Class"
                    confusion_matrix_elements.append(
                        ConfusionMatrixElement(str(class_name), ConfusionMatrixValue.Negative, float(confidence))
                    )
        else:
            ious = np.zeros((1, gt_labels.shape[0]))
        gts_detected = np.any(ious > threshold, axis=0)
        for k, gt_detection in enumerate(gts_detected):
            if not gt_detection:
                class_idx = int(gt_labels[k])
                class_name = names[class_idx] if 0 <= class_idx < len(names) else "Unknown Class"
                confusion_matrix_elements.append(
                    ConfusionMatrixElement(str(class_name), ConfusionMatrixValue.Positive, float(0))
                )
        if all(~gts_detected):
            confusion_matrix_elements.append(
                ConfusionMatrixElement("background", ConfusionMatrixValue.Positive, float(0))
            )
        confusion_matrices.append(confusion_matrix_elements)
    return confusion_matrices


@tensorleap_custom_metric("Confusion Matrix")
def confusion_matrix_metric(labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, targets: np.ndarray):
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    return confusion_matrix_metric_from_predictions(y_preds, targets)


@tensorleap_custom_metric("Confusion Matrix Concat Scores")
def confusion_matrix_metric_concat_scores(labels: np.ndarray, boxes_with_scores: np.ndarray, targets: np.ndarray):
    y_preds = format_rtdetr_concat_predictions(labels, boxes_with_scores)
    return confusion_matrix_metric_from_predictions(y_preds, targets)


@tensorleap_custom_metric("Confusion Matrix Class Scores")
def confusion_matrix_metric_class_scores(boxes_xyxy: np.ndarray, scores_per_class: np.ndarray, targets: np.ndarray):
    y_preds = format_class_scores_predictions(boxes_xyxy, scores_per_class)
    return confusion_matrix_metric_from_predictions(y_preds, targets)
