import cv2
import torch
import textwrap
import numpy as np
from typing import Dict, List

from utils.metrics import box_iou
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr
from leap_utils import compute_precision_recall_f1_fp_tp_fn
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.visualizers.default_visualizers import LeapImage
from utils.general import non_max_suppression, xyxy2xywh, xywh2xyxy
from code_loader.contract.enums import LeapDataType, MetricDirection, ConfusionMatrixValue
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.contract.datasetclasses import PreprocessResponse, SamplePreprocessResponse, ConfusionMatrixElement
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess, tensorleap_gt_encoder, tensorleap_input_encoder, tensorleap_custom_metric,
    tensorleap_metadata, tensorleap_custom_loss, tensorleap_custom_visualizer
)
from leap_utils import compute_iou, compute_accuracy
from leap_config import CONFIG, DATA_CONFIG, abs_path_from_root
from rtdetr_native.criterion import RTDETRCriterionv2
from rtdetr_native.matcher import HungarianMatcher

def format_rtdetr_predictions(labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).squeeze()
    boxes_xyxy = np.asarray(boxes_xyxy).squeeze()
    scores = np.asarray(scores).squeeze()

    if labels.ndim == 0:
        labels = np.array([labels], dtype=np.float32)
    if scores.ndim == 0:
        scores = np.array([scores], dtype=np.float32)
    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)

    score_threshold = float(CONFIG.get("score_threshold", 0.3))
    max_detections = int(CONFIG.get("max_detections", 300))
    keep = scores >= score_threshold
    labels = labels[keep]
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    if scores.size == 0:
        return np.zeros((1, 0, 6), dtype=np.float32)

    order = np.argsort(-scores)[:max_detections]
    labels = labels[order]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    pred = np.concatenate([boxes_xyxy, scores[:, None], labels[:, None]], axis=1).astype(np.float32)
    return pred[None, ...]


def _prediction_rows(y_preds: np.ndarray):
    y_preds = np.asarray(y_preds)
    if y_preds.ndim == 3 and y_preds.shape[-1] == 6:
        return [torch.from_numpy(y_preds[0].astype(np.float32))]
    return non_max_suppression(torch.from_numpy(y_preds))


def _label_names() -> List[str]:
    return DATA_CONFIG.get("pred_names", DATA_CONFIG.get("names", []))


COCO_CATEGORY_TO_LABEL = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
    17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40,
    47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53,
    60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66,
    77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


# ------------------------------
# Preprocessing and Encoders
# ------------------------------

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    """
    Loads datasets for 'train', 'val', and 'test' splits and wraps them in PreprocessResponse objects.

    Returns:
        List[PreprocessResponse]: List of datasets prepared for further processing.
    """
    data_yaml_path = abs_path_from_root(CONFIG["data_yaml_path"])
    data = check_dataset(data_yaml_path, autodownload=False)

    responses = []
    split_order = [split for split in ["train", "val", "test"] if split in data]
    if not split_order:
        raise ValueError(f"No supported splits found in dataset config: {data_yaml_path}")
    for split in split_order:
        _, dataset = create_dataloader(
                data[split],
                imgsz = CONFIG["image_size"],
                batch_size=1,
                stride=32,
                single_cls=False,
                rect=False,
                workers=1,
                prefix=colorstr(f"{split}: "),
                shuffle=False,
            )

        # Changed by KTH
        responses.append(PreprocessResponse(data=dataset, length=(len(dataset))))
        #responses.append(PreprocessResponse(data=dataset, length=100))
    return responses

@tensorleap_input_encoder('image', channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Retrieves and normalizes an image from the dataset.

    Args:
        idx (int): Index of the image.
        preprocess (PreprocessResponse): Dataset wrapper.

    Returns:
        np.ndarray: Normalized image array.
    """
    image = preprocess.data[idx][0].numpy().astype(np.float32)/255
    return image

@tensorleap_input_encoder('orig_size', channel_dim=1)
def input_size_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image_size = CONFIG["image_size"]
    return np.array([image_size, image_size], dtype=np.float32)

def _padded_gt_for_sample(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    mask = preprocessing.data.batch == idx
    img_size = preprocessing.data.img_size
    max_num_of_objs = int(CONFIG["max_num_of_objects"])
    labels_arr = []

    for i, is_selected in enumerate(mask):
        if not is_selected:
            continue
        labels = preprocessing.data.labels[i]  # shape: [N, 5] (label, x, y, w, h)
        cls = np.expand_dims(labels[:, 0], axis=1)
        x = np.expand_dims(labels[:, 1], axis=1)
        y = np.expand_dims(labels[:, 2], axis=1)
        w = np.expand_dims(labels[:, 3], axis=1)
        h = np.expand_dims(labels[:, 4], axis=1)

        if not preprocessing.data.rect:
            original_w, original_h = preprocessing.data.shapes[i]
            if original_w > original_h:
                new_h = original_h * img_size / original_w
                pad_size = img_size - new_h
                y = (y * new_h + (pad_size / 2)) / img_size
                h = h * new_h / img_size
            else:
                new_w = original_w * img_size / original_h
                pad_size = img_size - new_w
                x = (x * new_w + (pad_size / 2)) / img_size
                w = w * new_w / img_size

        adjusted = np.concatenate([cls, x, y, w, h], axis=1).astype(np.float32)
        if adjusted.shape[0] < max_num_of_objs:
            pad_rows = max_num_of_objs - adjusted.shape[0]
            pad = np.full((pad_rows, adjusted.shape[1]), -1, dtype=np.float32)
            adjusted = np.vstack([adjusted, pad])
        elif adjusted.shape[0] > max_num_of_objs:
            adjusted = adjusted[:max_num_of_objs, :]

        labels_arr.append(adjusted)

    if not labels_arr:
        return np.full((max_num_of_objs, 5), -1, dtype=np.float32)
    return np.array(labels_arr, dtype=np.float32).squeeze(0)


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    """
    Padded ground truth in [class, cx, cy, w, h] format.
    """
    return _padded_gt_for_sample(idx, preprocessing).astype(np.float32)


@tensorleap_gt_encoder("gt_boxes")
def gt_boxes_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    boxes = gt[:, 1:5].copy()
    invalid = (gt[:, 0] < 0)
    boxes[invalid] = 0.0
    return boxes.astype(np.float32)


@tensorleap_gt_encoder("gt_labels")
def gt_labels_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    return gt[:, 0].astype(np.float32)


@tensorleap_gt_encoder("gt_valid_mask")
def gt_valid_mask_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    valid = (gt[:, 0] >= 0).astype(np.float32)
    return valid

# ------------------------------
# Metadata
# ------------------------------
def average_dist_nn(boxes: np.array):
    """
    Computes the average distance to the nearest neigbour.

    Args:
        boxes (np.Array): An array of bounding boxes. Coordinates are normalized between 0 and 1.

    Returns:
        float: Average distance to the nearest neighbour.
    """
    if len(boxes) < 2:
        return 1.0
    data = boxes[:,:2]
    distance_matrix = np.full((len(data), len(data)), np.inf)
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            distance_matrix[i, j] = np.linalg.norm(data[i]-data[j])
    return float(np.mean(np.min(distance_matrix[:, 1:], axis=0)))

@tensorleap_metadata('metadata')
def sample_metadata(idx: int, preprocessing: PreprocessResponse) -> dict:
    """
    Computes the sample's metadata.

    Args:
        idx (int): Index of the image.
        preprocessing (PreprocessResponse): Dataset wrapper.

    Returns:
        dict: Dictionary with metadata values.
    """
    sample = preprocessing.data[idx]
    image = (sample[0].numpy().transpose(1,2,0)*255).astype(np.uint8)
    gt = sample[1].numpy()

    if gt.shape[0] != 0:
        gt_class = gt[:, 1]
        gt_bbox = gt[:,2:]
        bbox_areas = gt_bbox[:,2]*gt_bbox[:,3]
        bbox_cx = gt_bbox[:, 0]
        bbox_cy = gt_bbox[: 1]
        #nn_dist_mean = average_dist_nn(gt_bbox)
    else:
        gt_class, bbox_areas, bbox_cx, bbox_cy = np.array([]), np.array([]), np.array([]), np.array([])

    unique_classes, class_counts = np.unique(gt_class, return_counts=True)
    labels = DATA_CONFIG.get("pred_names", DATA_CONFIG.get("names", []))
    class_count_map = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, class_counts)}
    per_label_counts = {
        f"# of {label}": float(class_count_map[label_idx]) if label_idx in class_count_map else float(np.nan)
        for label_idx, label in enumerate(labels)
    }
    
    metadata_dict = {}

    # Zach Added
    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F) # Compute Laplacian variance
    sharpness = laplacian.var()
    metadata_dict.update({
        "image_sharpness": float(sharpness),
        "# of objects": gt.shape[0],
        "# of unique objects": len(unique_classes),
        "bbox area mean": float(bbox_areas.mean()),
        "bbox area median": float(np.median(bbox_areas)),
        "bbox area min": float(bbox_areas.min() if len(bbox_areas) > 0 else np.nan),
        "bbox area max": float(bbox_areas.max() if len(bbox_areas) > 0 else np.nan),
        "bbox area var": float(bbox_areas.var()),
        "bbox cx mean": float(bbox_cx.mean()),
        "bbox cx median": float(np.median(bbox_cx)),
        "bbox cx min": float(bbox_cx.min() if len(bbox_cx) > 0 else np.nan),
        "bbox cx max": float(bbox_cx.max() if len(bbox_cx) > 0 else np.nan),
        "bbox cx var": float(bbox_cx.var()),
        "bbox cy mean": float(bbox_cy.mean()),
        "bbox cy median": float(np.median(bbox_cy)),
        "bbox cy min": float(bbox_cy.min() if len(bbox_cy) > 0 else np.nan),
        "bbox cy max": float(bbox_cy.max() if len(bbox_cy) > 0 else np.nan),
        "bbox cy var": float(bbox_cy.var()),
        "bbox center var": float(bbox_cy.var()) + float(bbox_cx.var()), # a measure for object density
        #"average distance to NN": nn_dist_mean, # a measure for object density
        **per_label_counts,
    })
    return metadata_dict

# ------------------------------
# Custom Loss
# ------------------------------
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


def yolov5_loss_factory(num_scales):
    # Build predictions list
    preds_list = ', '.join([f'pred{i}' for i in range(num_scales)])
    all_args = f'{preds_list}, gt, demo_pred'

    # Dynamically generate function code
    fn_code = f'''
    @tensorleap_custom_loss("yolov5_loss")
    def yolov5_loss({all_args}):
        preds = [torch.from_numpy(p) for p in [{preds_list}]]
        gt = gt.squeeze(0)
        mask = ~(gt == -1).any(axis=1)
        # Filter out padding rows
        gt = gt[mask]
        gt_torch = torch.from_numpy(gt)
        gt_torch = torch.cat([torch.zeros_like(gt_torch[:, 1]).unsqueeze(1), gt_torch], dim=1)
        loss = yolov5_loss_compute(preds, gt_torch)[0]
        return loss.unsqueeze(0).numpy()
    '''
    local_ns = {}
    exec(textwrap.dedent(fn_code), globals(), local_ns)
    return local_ns['yolov5_loss']

# # new loss for our model
# @tensorleap_custom_loss("yolov5_new_loss")
# def yolov5_new_loss(pred0: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, pred3: np.ndarray, gt: np.ndarray, demo_pred: np.ndarray):
#     """
#     Computes YOLOv5-style object detection loss.
#
#     Args:
#         pred0, pred1, pred2 (np.ndarray): Prediction tensors for each detection scale.
#         gt (np.ndarray): Ground truth bounding boxes.
#         demo_pred (np.ndarray): Not used in loss computation. Added due to technical Tensorleap reason
#
#     Returns:
#         np.ndarray: Loss scalar.
#     """
#     loss = np.zeros(pred1.shape[0])
#     return loss

# new loss for our model
@tensorleap_custom_loss("yolov5_new_loss")
def yolov5_new_loss(pred0: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, pred3: np.ndarray, gt: np.ndarray, demo_pred: np.ndarray):
    """
    Computes YOLOv5-style object detection loss.

    Args:
        pred0, pred1, pred2 (np.ndarray): Prediction tensors for each detection scale.
        gt (np.ndarray): Ground truth bounding boxes.
        demo_pred (np.ndarray): Not used in loss computation. Added due to technical Tensorleap reason

    Returns:
        np.ndarray: Loss scalar.
    """
    loss = np.zeros(pred1.shape[0])
    return loss


# ------------------------------
# Visualizers
# ------------------------------

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    """
    Returns a LeapImage for visualization.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        LeapImage: Visualizable image object.
    """
    image = image.squeeze(0)
    image = image.transpose(1,2,0) # LeapImage visualizer expects inputs as channel last.
    return LeapImage((image*255).astype(np.uint8), compress=False)


@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays ground truth bounding boxes on the image.

    Args:
        image (np.ndarray): Input image.
        bb_gt (np.ndarray): Ground truth bounding boxes.

    Returns:
        LeapImageWithBBox: Image with bounding boxes drawn.
    """
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)  # LeapImageWithBBox visualizer expects inputs as channel last.
    image = (image*255).astype(np.uint8)

    bb_gt = bb_gt.squeeze(0)
    mask = ~(bb_gt == -1).any(axis=1)
    # Filter out padding rows
    bb_gt = bb_gt[mask]

    labels = _label_names()
    bboxes = []
    for bbx in bb_gt:
        label_idx = int(bbx[0]) if not np.isnan(bbx[0]) else -1
        if 0 <= label_idx < len(labels):
            label = labels[label_idx]
        else:
            label = "Unknown Class"
        bboxes.append(
            BoundingBox(
                x=bbx[1],
                y=bbx[2],
                width=bbx[3],
                height=bbx[4],
                confidence=1.0,
                label=label,
            )
        )
    return LeapImageWithBBox(data=image, bounding_boxes=bboxes)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, *, predictions: np.ndarray = None) -> LeapImageWithBBox:
    """
    Overlays predicted bounding boxes on the image after NMS and format conversion.

    Args:
        image (np.ndarray): Input image.
        predictions (np.ndarray): Raw prediction tensor.

    Returns:
        LeapImageWithBBox: Image with predicted bounding boxes.
    """
    if predictions is None:
        predictions = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    prediction_rows = _prediction_rows(predictions)
    preds = prediction_rows[0].numpy() if len(prediction_rows) > 0 else np.zeros((0, 6), dtype=np.float32)
    preds = xyxy2xywh(preds)

    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)  # LeapImageWithBBox visualizer expects inputs as channel last.
    image = (image * 255).astype(np.uint8)
    h, w, _ = image.shape

    label_names = _label_names()
    bboxes = []
    for pred in preds:
        label_idx = int(pred[5]) if not np.isnan(pred[5]) else -1
        if 0 <= label_idx < len(label_names):
            label = label_names[label_idx]
        else:
            label = "Unknown Class"
        bboxes.append(
            BoundingBox(
                x=pred[0] / w,
                y=pred[1] / h,
                width=pred[2] / w,
                height=pred[3] / h,
                confidence=pred[4],
                label=label,
            )
        )
    return LeapImageWithBBox(data=image, bounding_boxes=bboxes)

# ------------------------------
# Custom Metrics
# ------------------------------

@tensorleap_custom_metric(name="per_sample_metrics", direction={
            "precision": MetricDirection.Upward,
            "recall": MetricDirection.Upward,
            "f1": MetricDirection.Upward,
            "FP": MetricDirection.Downward,
            "TP": MetricDirection.Upward,
            "FN": MetricDirection.Downward,
            "iou": MetricDirection.Upward,
            "accuracy": MetricDirection.Upward,
        })
def get_per_sample_metrics(labels, boxes_xyxy, scores, targets: np.ndarray):
    """
    Calculates metrics per sample on the model's prediction

    Args:
        y_pred (np.ndarray): Prediction from model.
        targets (np.ndarray): Ground truth.

    Returns:
        dict: Dictionary with metric values.
    """
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)
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
    preds = _prediction_rows(y_preds)
    for pred, gt in zip(preds, targets):

        mask = ~(gt == -1).any(axis=1)
        # Filter out padding rows
        gt = gt[mask]
        gt = torch.from_numpy(gt)


        if gt.shape[0] == 0 and pred.shape[0] == 0:
            _update_metrics(metrics,np.nan, np.nan, 0, 0, 0, 0, 1, 1) # Edge case: no objects, assume perfect
            continue

        if pred.shape[0] == 0:
            _update_metrics(metrics, np.nan, 0, 0, 0, 0, gt.shape[0], 0, 0)  # No predictions at all
            continue

        if gt.shape[0] == 0:
            _update_metrics(metrics, 0, np.nan, 0, pred.shape[0], 0, 0, 0, 0) # No GT but has predictions
            continue

        pred_boxes = pred[:, :4] / CONFIG["image_size"] # normalize to be [0,1]
        pred_labels = pred[:, 5]

        gt_boxes = xywh2xyxy(gt[:, 1:])
        gt_labels = gt[:, 0]

        p, r, f1, FP, TP, FN = compute_precision_recall_f1_fp_tp_fn(gt_boxes, pred_boxes, iou_threshold=0.1)
        iou = compute_iou(gt_boxes, pred_boxes)
        acc = compute_accuracy(gt_boxes, gt_labels, pred_boxes, pred_labels)
        _update_metrics(metrics, float(p), float(r), float(f1), int(FP), int(TP), int(FN), float(iou), float(acc))
    return metrics

@tensorleap_custom_metric('Confusion Matrix')
def confusion_matrix_metric(labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, targets: np.ndarray):
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    threshold=0.1
    confusion_matrices = []
    label_names = _label_names()
    preds = _prediction_rows(y_preds)
    for pred, gt in zip(preds, targets):
        confusion_matrix_elements = []

        mask = ~(gt == -1).any(axis=1)
        # Filter out padding rows
        gt = gt[mask]
        gt = torch.from_numpy(gt)
        gt_bbox = xywh2xyxy(gt[:, 1:])
        gt_labels = gt[:, 0]

        pred_boxes = pred[:, :4] / CONFIG["image_size"]  # normalize to be [0,1]

        if pred.shape[0] != 0 and gt_bbox.shape[0] != 0:
            ious = box_iou(gt_bbox, pred_boxes).numpy().T
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt_labels[max_iou_ind[i]])
                class_name = label_names[gt_idx] if 0 <= gt_idx < len(label_names) else "Unknown Class"
                gt_label = f"{class_name}"
                confidence = pred[i, 4]
                if prediction:  # TP
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else:  # FP
                    pred_idx = int(pred[i, 5])
                    class_name = label_names[pred_idx] if 0 <= pred_idx < len(label_names) else "Unknown Class"
                    pred_label = f"{class_name}"
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        else:  # No prediction
            ious = np.zeros((1, gt_labels.shape[0]))
        gts_detected = np.any((ious > threshold), axis=0)
        for k, gt_detection in enumerate(gts_detected):
            label_idx = gt_labels[k]
            if not gt_detection : # FN
                class_idx = int(label_idx)
                class_name = label_names[class_idx] if 0 <= class_idx < len(label_names) else "Unknown Class"
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    f"{class_name}",
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        if all(~ gts_detected):
            confusion_matrix_elements.append(ConfusionMatrixElement(
                "background",
                ConfusionMatrixValue.Positive,
                float(0)
            ))
        confusion_matrices.append(confusion_matrix_elements)
    return confusion_matrices
