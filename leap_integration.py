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
    confusion_matrix_metric,
    detection_f1_loss,
    detection_iou_loss,
    get_per_sample_metrics,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    input_encoder,
    preprocess_func_leap,
    pred_bb_decoder,
    sample_metadata,
)
from leap_config import CONFIG, abs_path_from_root


PREDICTION_TYPES = [
    PredictionTypeHandler(name="boxes", labels=["x1", "y1", "x2", "y2"], channel_dim=-1),
    PredictionTypeHandler(name="scores", labels=[], channel_dim=-1),
]


@tensorleap_load_model(PREDICTION_TYPES)
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    if not model_path.endswith(".onnx"):
        raise ValueError("Only ONNX is supported in this integration file.")
    session = ort.InferenceSession(model_path)
    model_input = session.get_inputs()[0]
    CONFIG["_model_input_name"] = model_input.name
    CONFIG["_model_input_hw"] = [model_input.shape[2], model_input.shape[3]]
    return session


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)
    _ = gt_boxes_encoder(idx, subset)
    _ = gt_labels_encoder(idx, subset)
    _ = gt_valid_mask_encoder(idx, subset)

    input_name = model.get_inputs()[0].name
    predictions = model.run(None, {input_name: image})

    boxes_output = predictions[0]
    scores_output = predictions[1]

    vis_image = image_visualizer(image)
    vis_gt = bb_decoder(image, gt, boxes_output, scores_output)
    vis_pred = pred_bb_decoder(image, boxes_output, scores_output)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    _ = get_per_sample_metrics(boxes_output, scores_output, gt)
    _ = confusion_matrix_metric(boxes_output, scores_output, gt)
    _ = detection_iou_loss(boxes_output, scores_output, gt)
    _ = detection_f1_loss(boxes_output, scores_output, gt)
    _ = sample_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    sample_idx = int(CONFIG.get("check_sample_index", 0))
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise IndexError(f"check_subset_index {subset_idx} out of range, {len(subsets)} subsets available")
    check_integration(sample_idx, subsets[subset_idx])
