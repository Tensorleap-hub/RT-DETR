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
    bb_decoder_class_scores,
    confusion_matrix_metric_class_scores,
    detection_f1_loss_class_scores,
    detection_iou_loss_class_scores,
    get_per_sample_metrics_class_scores,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    input_encoder,
    preprocess_func_leap,
    pred_bb_decoder_class_scores,
    sample_metadata,
)
from leap_config import CONFIG, abs_path_from_root


OUTPUT_INDICES = {
    "boxes": int(CONFIG.get("output_indices", {}).get("boxes", 0)),
    "scores": int(CONFIG.get("output_indices", {}).get("scores", 1)),
    "logits": int(CONFIG.get("output_indices", {}).get("logits", 2)),
}

PREDICTION_TYPES = [
    PredictionTypeHandler(name="boxes", labels=["x1", "y1", "x2", "y2"], channel_dim=-1),
    PredictionTypeHandler(name="scores", labels=[], channel_dim=-1),
    PredictionTypeHandler(name="logits", labels=[], channel_dim=-1),
]


@tensorleap_load_model(PREDICTION_TYPES)
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    if not model_path.endswith(".onnx"):
        raise ValueError("Only ONNX is supported in this integration file.")
    return ort.InferenceSession(model_path)


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)
    _ = gt_boxes_encoder(idx, subset)
    _ = gt_labels_encoder(idx, subset)
    _ = gt_valid_mask_encoder(idx, subset)

    predictions = model.run(None, {"images": image})

    boxes_output = predictions[OUTPUT_INDICES["boxes"]]
    scores = predictions[OUTPUT_INDICES["scores"]]

    vis_image = image_visualizer(image)
    vis_gt = bb_decoder_class_scores(image, gt, boxes_output, scores)
    vis_pred = pred_bb_decoder_class_scores(image, boxes_output, scores)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    _ = get_per_sample_metrics_class_scores(boxes_output, scores, gt)
    _ = confusion_matrix_metric_class_scores(boxes_output, scores, gt)
    _ = detection_iou_loss_class_scores(boxes_output, scores, gt)
    _ = detection_f1_loss_class_scores(boxes_output, scores, gt)
    _ = sample_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    sample_idx = int(CONFIG.get("check_sample_index", 0))
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise IndexError(f"check_subset_index {subset_idx} out of range, {len(subsets)} subsets available")
    check_integration(sample_idx, subsets[subset_idx])
