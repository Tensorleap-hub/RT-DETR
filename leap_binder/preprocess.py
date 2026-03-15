from typing import List

import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_preprocess,
)
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr

from .common import CONFIG, abs_path_from_root


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    data_yaml_path = abs_path_from_root(CONFIG["data_yaml_path"])
    data = check_dataset(data_yaml_path, autodownload=False)

    responses = []
    split_order = [split for split in ["train", "val", "test"] if split in data]
    if not split_order:
        raise ValueError(f"No supported splits found in dataset config: {data_yaml_path}")
    for split in split_order:
        _, dataset = create_dataloader(
            data[split],
            imgsz=CONFIG["image_size"],
            batch_size=1,
            stride=32,
            single_cls=False,
            rect=False,
            workers=1,
            prefix=colorstr(f"{split}: "),
            shuffle=False,
        )
        responses.append(PreprocessResponse(data=dataset, length=len(dataset)))
    return responses


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image = preprocess.data[idx][0].numpy().astype(np.float32) / 255
    return image


@tensorleap_input_encoder("orig_size", channel_dim=1)
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
        labels = preprocessing.data.labels[i]
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


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return _padded_gt_for_sample(idx, preprocessing).astype(np.float32)


@tensorleap_gt_encoder("gt_boxes")
def gt_boxes_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    gt = _padded_gt_for_sample(idx, preprocessing)
    boxes = gt[:, 1:5].copy()
    invalid = gt[:, 0] < 0
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
