import onnxruntime as ort
import numpy as np
import os
os.environ['TL_DISABLE_ANALYTICS'] = 'true'
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from leap_binder import (
    bb_decoder,
    bb_decoder_class_scores,
    bb_decoder_concat_scores,
    confusion_matrix_metric,
    confusion_matrix_metric_class_scores,
    confusion_matrix_metric_concat_scores,
    detection_f1_loss,
    detection_f1_loss_class_scores,
    detection_f1_loss_concat_scores,
    detection_iou_loss,
    detection_iou_loss_class_scores,
    detection_iou_loss_concat_scores,
    get_per_sample_metrics,
    get_per_sample_metrics_class_scores,
    get_per_sample_metrics_concat_scores,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    input_encoder,
    input_size_encoder,
    pred_bb_decoder,
    pred_bb_decoder_class_scores,
    pred_bb_decoder_concat_scores,
    preprocess_func_leap,
    rtdetr_loss_components_native,
    rtdetr_total_loss_native,
    sample_metadata,
)
from leap_config import CONFIG, DATA_CONFIG, abs_path_from_root


LABEL_NAMES = DATA_CONFIG.get("pred_names", DATA_CONFIG.get("names", []))
MODEL_OUTPUT_FORMAT = str(CONFIG.get("model_output_format", "rtdetr_raw"))
MODEL_HAS_SEPARATE_SCORES = MODEL_OUTPUT_FORMAT in {"rtdetr_raw", "detections"}
MODEL_HAS_CONCAT_SCORES = MODEL_OUTPUT_FORMAT == "detections_concat_scores"
MODEL_HAS_RAW_PREDICTIONS = MODEL_OUTPUT_FORMAT == "rtdetr_raw"
MODEL_HAS_CLASS_SCORES = MODEL_OUTPUT_FORMAT == "class_scores"


def _output_index(name: str, default: int) -> int:
    return int(CONFIG.get("output_indices", {}).get(name, default))


if MODEL_HAS_CLASS_SCORES:
    OUTPUT_INDICES = {
        "boxes": _output_index("boxes", 0),
        "scores": _output_index("scores", 1),
        "logits": _output_index("logits", 2),
    }
else:
    OUTPUT_INDICES = {
        "labels": _output_index("labels", 0),
        "boxes": _output_index("boxes", 1),
    }
    if MODEL_HAS_SEPARATE_SCORES:
        OUTPUT_INDICES["scores"] = _output_index("scores", 2)
    if MODEL_HAS_RAW_PREDICTIONS:
        OUTPUT_INDICES["pred_logits"] = _output_index("pred_logits", 3)
        OUTPUT_INDICES["pred_boxes"] = _output_index("pred_boxes", 4)

if MODEL_HAS_CLASS_SCORES:
    PREDICTION_TYPES = [
        PredictionTypeHandler(name="boxes", labels=["x1", "y1", "x2", "y2"], channel_dim=-1),
        PredictionTypeHandler(name="scores", labels=LABEL_NAMES, channel_dim=-1),
        PredictionTypeHandler(name="logits", labels=LABEL_NAMES, channel_dim=-1),
    ]
    SELECTED_BB_DECODER = bb_decoder_class_scores
    SELECTED_PRED_BB_DECODER = pred_bb_decoder_class_scores
    SELECTED_PER_SAMPLE_METRIC = get_per_sample_metrics_class_scores
    SELECTED_CONFUSION_MATRIX_METRIC = confusion_matrix_metric_class_scores
    SELECTED_IOU_LOSS = detection_iou_loss_class_scores
    SELECTED_F1_LOSS = detection_f1_loss_class_scores
elif MODEL_HAS_SEPARATE_SCORES:
    PREDICTION_TYPES = [
        PredictionTypeHandler(name="labels", labels=LABEL_NAMES, channel_dim=-1),
        PredictionTypeHandler(name="boxes", labels=["x1", "y1", "x2", "y2"], channel_dim=-1),
        PredictionTypeHandler(name="confidence", labels=["score"], channel_dim=-1),
    ]
    if MODEL_HAS_RAW_PREDICTIONS:
        PREDICTION_TYPES += [
            PredictionTypeHandler(name="pred_logits", labels=LABEL_NAMES, channel_dim=-1),
            PredictionTypeHandler(name="pred_boxes", labels=["cx", "cy", "w", "h"], channel_dim=-1),
        ]
    SELECTED_BB_DECODER = bb_decoder
    SELECTED_PRED_BB_DECODER = pred_bb_decoder
    SELECTED_PER_SAMPLE_METRIC = get_per_sample_metrics
    SELECTED_CONFUSION_MATRIX_METRIC = confusion_matrix_metric
    SELECTED_IOU_LOSS = detection_iou_loss
    SELECTED_F1_LOSS = detection_f1_loss
else:
    PREDICTION_TYPES = [
        PredictionTypeHandler(name="labels", labels=LABEL_NAMES, channel_dim=-1),
        PredictionTypeHandler(name="boxes", labels=["x1", "y1", "x2", "y2", "score"], channel_dim=-1),
    ]
    SELECTED_BB_DECODER = bb_decoder_concat_scores
    SELECTED_PRED_BB_DECODER = pred_bb_decoder_concat_scores
    SELECTED_PER_SAMPLE_METRIC = get_per_sample_metrics_concat_scores
    SELECTED_CONFUSION_MATRIX_METRIC = confusion_matrix_metric_concat_scores
    SELECTED_IOU_LOSS = detection_iou_loss_concat_scores
    SELECTED_F1_LOSS = detection_f1_loss_concat_scores


@tensorleap_load_model(PREDICTION_TYPES)
def load_model():
    """
    Load the trained model for inference.
    """
    model_path = abs_path_from_root(CONFIG["model_path"])
    if not model_path.endswith(".onnx"):
        raise ValueError("Only ONNX is supported in this integration file.")
    return ort.InferenceSession(model_path)


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)
    gt_boxes = gt_boxes_encoder(idx, subset)
    gt_labels = gt_labels_encoder(idx, subset)
    gt_valid_mask = gt_valid_mask_encoder(idx, subset)
    orig_sizes = input_size_encoder(idx, subset)

    image_for_model = np.expand_dims(image, axis=0) if image.ndim == 3 else image

    if MODEL_HAS_CLASS_SCORES:
        predictions = model.run(None, {"images": image_for_model})
    else:
        if idx is None:
            model_inputs = {"images": image, "orig_target_sizes": orig_sizes}
        else:
            orig_sizes_for_model = orig_sizes.astype(np.int64) if isinstance(orig_sizes, np.ndarray) else orig_sizes
            if isinstance(orig_sizes_for_model, np.ndarray) and orig_sizes_for_model.ndim == 1:
                orig_sizes_for_model = np.expand_dims(orig_sizes_for_model, axis=0)
            model_inputs = {"images": image_for_model, "orig_target_sizes": orig_sizes_for_model}
        predictions = model.run(None, model_inputs)

    if MODEL_HAS_CLASS_SCORES:
        boxes_output = predictions[OUTPUT_INDICES["boxes"]]
        scores = predictions[OUTPUT_INDICES["scores"]]
        prediction_args = (boxes_output, scores)
    else:
        labels = predictions[OUTPUT_INDICES["labels"]]
        boxes_output = predictions[OUTPUT_INDICES["boxes"]]
        prediction_args = (labels, boxes_output)
        if MODEL_HAS_SEPARATE_SCORES:
            scores = predictions[OUTPUT_INDICES["scores"]]
            prediction_args = (labels, boxes_output, scores)

    vis_image = image_visualizer(image)
    vis_gt = SELECTED_BB_DECODER(image, gt, *prediction_args)
    vis_pred = SELECTED_PRED_BB_DECODER(image, *prediction_args)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    _ = SELECTED_PER_SAMPLE_METRIC(*prediction_args, gt)
    _ = SELECTED_CONFUSION_MATRIX_METRIC(*prediction_args, gt)
    _ = SELECTED_IOU_LOSS(*prediction_args, gt)
    _ = SELECTED_F1_LOSS(*prediction_args, gt)
    if MODEL_HAS_RAW_PREDICTIONS:
        pred_logits = predictions[OUTPUT_INDICES["pred_logits"]]
        pred_boxes = predictions[OUTPUT_INDICES["pred_boxes"]]
        _ = rtdetr_total_loss_native(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid_mask)
        _ = rtdetr_loss_components_native(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid_mask)
    _ = sample_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    sample_idx = int(CONFIG.get("check_sample_index", 0))
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise IndexError(f"check_subset_index out of range: {subset_idx}, available subsets: {len(subsets)}")
    check_integration(sample_idx, subsets[subset_idx])
