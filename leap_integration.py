import onnxruntime as ort
import numpy as np

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from leap_binder import *
from leap_config import CONFIG, DATA_CONFIG, abs_path_from_root


prediction_type = PredictionTypeHandler(
    name="labels",
    labels=DATA_CONFIG["pred_names"],
    channel_dim=-1,
)
prediction_type1 = PredictionTypeHandler(
    name="boxes",
    labels=["x", "y", "w", "h"],
    channel_dim=-1,
)

prediction_type2 = PredictionTypeHandler(
    name="confidence",
    labels=["obj_conf"],
    channel_dim=-1,
)


@tensorleap_load_model([prediction_type,prediction_type1,prediction_type2])
def load_model():
    """
    Load the trained model for inference.
    TODO: Update CONFIG['model_path'] with the real ONNX model path.
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

    model_inputs = model.get_inputs()
    if CONFIG['model_type'] == "RTDETR":
        image_size = input_size_encoder(idx, subset)
        predictions = model.run(
            None,
            {
                "images": image,
                "orig_target_sizes": image_size,
            },
        )
        labels, boxes_xyxy, scores = predictions[0], predictions[1], predictions[2]
    else:
        input_name = model_inputs[0].name
        predictions = model.run(None, {input_name: image})
        main_pred = predictions[0]

    vis_image = image_visualizer(image)
    vis_gt = gt_bb_decoder(image, gt)
    vis_pred = bb_decoder(image, labels, boxes_xyxy, scores)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    _ = get_per_sample_metrics(labels, boxes_xyxy, scores, gt)
    _ = confusion_matrix_metric(labels, boxes_xyxy, scores, gt)
    _ = sample_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    sample_idx = int(CONFIG.get("check_sample_index", 0))
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise IndexError(f"check_subset_index out of range: {subset_idx}, available subsets: {len(subsets)}")
    check_integration(sample_idx, subsets[subset_idx])
