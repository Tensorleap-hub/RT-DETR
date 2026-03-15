import numpy as np

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer
from code_loader.visualizers.default_visualizers import LeapImage
from utils.general import xyxy2xywh

from .common import format_rtdetr_predictions, label_names, prediction_rows


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    image = image.transpose(1, 2, 0)
    return (image * 255).astype(np.uint8)


def _squeeze_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    if boxes.ndim == 3:
        return boxes[0]
    return boxes


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    return LeapImage(_image_to_uint8(image), compress=False)


@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    image_data = _image_to_uint8(image)
    bb_gt = _squeeze_boxes(bb_gt)
    mask = ~(bb_gt == -1).any(axis=1)
    bb_gt = bb_gt[mask]

    labels = label_names()
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
    return LeapImageWithBBox(data=image_data, bounding_boxes=bboxes)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(
    image: np.ndarray,
    labels: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    *,
    predictions: np.ndarray = None,
) -> LeapImageWithBBox:
    if predictions is None:
        predictions = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    prediction_tensor = prediction_rows(predictions)
    preds = prediction_tensor[0].numpy() if len(prediction_tensor) > 0 else np.zeros((0, 6), dtype=np.float32)
    preds = xyxy2xywh(preds)

    image_data = _image_to_uint8(image)
    h, w, _ = image_data.shape

    names = label_names()
    bboxes = []
    for pred in preds:
        label_idx = int(pred[5]) if not np.isnan(pred[5]) else -1
        if 0 <= label_idx < len(names):
            label = names[label_idx]
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
    return LeapImageWithBBox(data=image_data, bounding_boxes=bboxes)
