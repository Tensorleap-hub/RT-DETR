import numpy as np

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer
from code_loader.visualizers.default_visualizers import LeapImage
from utils.general import xyxy2xywh

from .common import CONFIG, format_predictions, label_names, prediction_rows


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    elif image.shape[-1] not in (1, 3):
        raise ValueError(f"Unsupported image shape for visualization: {image.shape}")
    if image.dtype == np.uint8:
        return image
    return (image * 255).astype(np.uint8)


def _squeeze_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.ndim == 3:
        return boxes[0]
    return boxes


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    return LeapImage(_image_to_uint8(image), compress=False)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(
    image: np.ndarray,
    bb_gt: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
) -> LeapImageWithBBox:
    predictions = format_predictions(boxes_xyxy, scores_per_class)
    return _bb_decoder_from_predictions(image, bb_gt, predictions)


@tensorleap_custom_visualizer("pred_bb_decoder", LeapDataType.ImageWithBBox)
def pred_bb_decoder(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
) -> LeapImageWithBBox:
    predictions = format_predictions(boxes_xyxy, scores_per_class)
    return _pred_bb_decoder_from_predictions(image, predictions)


def _bb_decoder_from_predictions(
    image: np.ndarray,
    bb_gt: np.ndarray,
    predictions: np.ndarray,
) -> LeapImageWithBBox:
    image_data = _image_to_uint8(image)
    bb_gt = _squeeze_boxes(bb_gt)
    mask = ~(bb_gt == -1).any(axis=1)
    bb_gt = bb_gt[mask]

    labels = label_names()
    bboxes = []
    for bbx in bb_gt:
        label_idx = int(bbx[0]) if not np.isnan(bbx[0]) else -1
        if 0 <= label_idx < len(labels):
            label = labels[label_idx] + "_GT"
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
    pred_boxes = _pred_bb_creator(image, predictions=predictions)
    bboxes = bboxes + pred_boxes
    return LeapImageWithBBox(data=image_data, bounding_boxes=bboxes)


def _pred_bb_decoder_from_predictions(
    image: np.ndarray,
    predictions: np.ndarray,
) -> LeapImageWithBBox:
    bboxes = _pred_bb_creator(image, predictions=predictions)
    return LeapImageWithBBox(data=_image_to_uint8(image), bounding_boxes=bboxes)


def _pred_bb_creator(
    image: np.ndarray,
    *,
    predictions: np.ndarray,
) -> list[BoundingBox]:
    prediction_tensor = prediction_rows(predictions)
    preds = prediction_tensor[0].numpy() if len(prediction_tensor) > 0 else np.zeros((0, 6), dtype=np.float32)

    image_data = _image_to_uint8(image)
    h, w, _ = image_data.shape

    pred_fmt = CONFIG.get("pred_bbox_format", "xyxy_abs")
    if pred_fmt == "cxcywh_norm":
        boxes_norm = preds[:, :4]
    else:
        raw = xyxy2xywh(preds)
        boxes_norm = raw[:, :4] / np.array([w, h, w, h], dtype=np.float32)

    names = label_names()
    bboxes = []
    for i, pred in enumerate(preds):
        label_idx = int(pred[5]) if not np.isnan(pred[5]) else -1
        if 0 <= label_idx < len(names):
            label = names[label_idx] + "_PRED"
        else:
            label = "Unknown Class_PRED"
        bx = boxes_norm[i]
        bboxes.append(
            BoundingBox(
                x=float(bx[0]),
                y=float(bx[1]),
                width=float(bx[2]),
                height=float(bx[3]),
                confidence=pred[4],
                label=label,
            )
        )
    return bboxes
