import onnxruntime as ort
import numpy as np

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from leap_binder import (
    pred_bb_decoder,
    confusion_matrix_metric,
    get_per_sample_metrics,
    bb_decoder,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    input_encoder,
    input_size_encoder,
    preprocess_func_leap,
    rtdetr_loss_components_native,
    rtdetr_total_loss_native,
    sample_metadata,
)
from leap_config import CONFIG, DATA_CONFIG, abs_path_from_root


LABEL_NAMES = DATA_CONFIG.get("pred_names", DATA_CONFIG.get("names", []))
OUTPUT_INDICES = {
    "labels": int(CONFIG.get("output_indices", {}).get("labels", 0)),
    "boxes": int(CONFIG.get("output_indices", {}).get("boxes", 1)),
    "scores": int(CONFIG.get("output_indices", {}).get("scores", 2)),
    "pred_logits": int(CONFIG.get("output_indices", {}).get("pred_logits", 3)),
    "pred_boxes": int(CONFIG.get("output_indices", {}).get("pred_boxes", 4)),
}


prediction_type = PredictionTypeHandler(
    name="labels",
    labels=LABEL_NAMES,
    channel_dim=-1,
)
prediction_type1 = PredictionTypeHandler(
    name="boxes",
    labels=["x1", "y1", "x2", "y2"],
    channel_dim=-1,
)

prediction_type2 = PredictionTypeHandler(
    name="confidence",
    labels=["score"],
    channel_dim=-1,
)

prediction_type3 = PredictionTypeHandler(
    name="pred_logits",
    labels=LABEL_NAMES,
    channel_dim=-1,
)

prediction_type4 = PredictionTypeHandler(
    name="pred_boxes",
    labels=["cx", "cy", "w", "h"],
    channel_dim=-1,
)


@tensorleap_load_model([prediction_type, prediction_type1, prediction_type2, prediction_type3, prediction_type4])
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

    if idx is None:
        predictions = model.run(
            None,
            {
                "images": image,
                "orig_target_sizes": orig_sizes,
            },
        )
    else:
        image_for_model = np.expand_dims(image, axis=0) if image.ndim == 3 else image
        if isinstance(orig_sizes, np.ndarray):
            orig_sizes_for_model = orig_sizes.astype(np.int64)
            if orig_sizes_for_model.ndim == 1:
                orig_sizes_for_model = np.expand_dims(orig_sizes_for_model, axis=0)
        else:
            orig_sizes_for_model = orig_sizes
        predictions = model.run(
            None,
            {
                "images": image_for_model,
                "orig_target_sizes": orig_sizes_for_model,
            },
        )

    labels = predictions[OUTPUT_INDICES["labels"]]
    boxes_xyxy = predictions[OUTPUT_INDICES["boxes"]]
    scores = predictions[OUTPUT_INDICES["scores"]]

    vis_image = image_visualizer(image)
    vis_gt = bb_decoder(image, gt,  labels, boxes_xyxy, scores)
    vis_pred = pred_bb_decoder(image, labels, boxes_xyxy, scores)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    _ = get_per_sample_metrics(labels, boxes_xyxy, scores, gt)
    _ = confusion_matrix_metric(labels, boxes_xyxy, scores, gt)
    pred_logits_idx = OUTPUT_INDICES["pred_logits"]
    pred_boxes_idx = OUTPUT_INDICES["pred_boxes"]
    pred_logits = predictions[pred_logits_idx]
    pred_boxes = predictions[pred_boxes_idx]
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
