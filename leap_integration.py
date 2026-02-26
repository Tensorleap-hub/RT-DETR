import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.helpers.visualizer.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from leap_binder import (
    bb_decoder,
    confusion_matrix_metric,
    get_per_sample_metrics,
    gt_bb_decoder,
    gt_encoder,
    image_visualizer,
    input_encoder,
    preprocess_func_leap,
    sample_metadata,
    yolov5_loss,
)
from leap_config import CONFIG, DATA_CONFIG, abs_path_from_root


prediction_type = PredictionTypeHandler(
    name="object detection",
    labels=["x", "y", "w", "h", "obj_conf"] + DATA_CONFIG["pred_names"],
    channel_dim=-1,
)


@tensorleap_load_model([prediction_type])
def load_model():
    """
    Load the trained model for inference.
    TODO: Update CONFIG['model_path'] with the real ONNX model path.
    """
    model_path = abs_path_from_root(CONFIG["model_path"])
    if CONFIG.get("model_type", "onnx") != "onnx":
        raise ValueError("Only ONNX is supported in this integration file.")
    return ort.InferenceSession(model_path)


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)

    input_name = model.get_inputs()[0].name
    predictions = model.run(None, {input_name: image})
    main_pred = predictions[0]
    anchor_preds = predictions[1:]

    vis_image = image_visualizer(image)
    vis_gt = gt_bb_decoder(image, gt)
    vis_pred = bb_decoder(image, main_pred)

    _ = vis_image
    _ = vis_gt
    _ = vis_pred
    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt, title="Ground truth boxes")
        visualize(vis_pred, title="Predicted boxes")

    if len(anchor_preds) < 4:
        raise ValueError(
            f"Expected at least 5 model outputs (1 main + 4 anchors), got {len(predictions)}."
        )

    _ = yolov5_loss(anchor_preds[0], anchor_preds[1], anchor_preds[2], anchor_preds[3], gt, main_pred)
    _ = get_per_sample_metrics(main_pred, gt)
    _ = confusion_matrix_metric(main_pred, gt)
    _ = sample_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    sample_idx = int(CONFIG.get("check_sample_index", 0))
    if subset_idx < 0 or subset_idx >= len(subsets):
        raise IndexError(f"check_subset_index out of range: {subset_idx}, available subsets: {len(subsets)}")
    check_integration(sample_idx, subsets[subset_idx])
