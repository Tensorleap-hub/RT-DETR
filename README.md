# RT-DETR Tensorleap Integration

This repository contains a Tensorleap integration for object detection using an ONNX RT-DETR model. The current integration entrypoint is `leap_integration.py`, with dataset preprocessing, metrics, visualizers, losses, and metadata implemented under `leap_binder/`.

The integration is dataset-configurable from `leap_config.yaml` and currently supports:

- `coco`
- `coco128`
- `visdrone128`
- `visdrone`

The default out-of-the-box configuration uses the committed `visdrone128` subset together with `rtdetrv2_r18vd_120e_raw_outputs.onnx`, so a fresh clone can run local validation without downloading a dataset first.

## Prerequisites

- Python `>=3.9,<3.11`
- Tensorleap CLI installed and authenticated if you plan to push to the platform

## Installation

### Poetry

```bash
poetry install
```

Run local commands with:

```bash
poetry run python leap_integration.py
```

### pip

If you want a local pip-based environment instead of Poetry, use:

```bash
pip install -r local_requirements.txt
```

`local_requirements.txt` mirrors the dependencies from `pyproject.toml` for local use.

Do not use `requirements.txt` for local setup. In this repository, `requirements.txt` is reserved for Tensorleap packaging via `leap.yaml`.

## Configuration

Main configuration lives in `leap_config.yaml`.

### Dataset selection

Switch datasets by changing:

```yaml
dataset_name: "coco"
```

The default value in this repository is:

```yaml
dataset_name: "visdrone128"
```

The dataset root path is configured separately in `leap_config.yaml`:

```yaml
dataset_path:
  - "data/visdrone128"
  - "/data/visdrone128"
```

`dataset_path` may be either a single root path or an ordered list of candidate roots. The integration checks the candidates in order and uses the first one that contains the configured dataset split paths. If none of the candidates contain the dataset, it raises an exception listing the attempted roots.

Supported values:

- `coco`
- `coco128`
- `visdrone128`
- `visdrone`

Dataset profiles are resolved in `leap_config.py` and point to:

- `data/coco.yaml`
- `data/coco128.yaml`
- `data/visdrone128.yaml`
- `data/visdrone.yaml`

The dataset YAML selects split structure and class names. The actual filesystem root is taken from `dataset_path` in `leap_config.yaml`, not from the dataset YAML.

### Dataset autodownload

- `visdrone` defaults to `dataset_autodownload: true`
- `coco`, `coco128`, and `visdrone128` default to `dataset_autodownload: false`

You can still override `dataset_autodownload` explicitly in `leap_config.yaml` if needed.

### Model configuration

The current active model is configured via:

```yaml
model_path: "rtdetrv2_r18vd_120e_raw_outputs.onnx"
```

The current integration expects an ONNX model with:

- inputs:
  - `images`
  - `orig_target_sizes`
- outputs:
  - `labels`
  - `boxes`
  - `scores`
  - `pred_logits`
  - `pred_boxes`

If you change the ONNX model contract, the integration code may need to change as well.

## Local validation

Run the integration locally before pushing:

```bash
poetry run python leap_integration.py
```

This validates:

- dataset loading
- model inference
- visualizers
- custom metrics
- metadata
- custom loss hooks

## Project structure

Key files and folders:

- `leap_integration.py`: Tensorleap entrypoint and integration test
- `leap_config.yaml`: project configuration
- `leap_config.py`: config loader and dataset profile resolution
- `data/`: dataset YAML definitions
- `leap_binder/preprocess.py`: preprocess and encoders
- `leap_binder/metrics.py`: custom metrics
- `leap_binder/visualizers.py`: image and bounding box visualizers
- `leap_binder/losses.py`: custom loss and RT-DETR loss components
- `leap_binder/metadata.py`: per-sample metadata

## Dataset notes

### VisDrone

`data/visdrone.yaml` downloads the VisDrone detection archives and converts annotations into YOLO-format label files during setup.

### VisDrone128

`data/visdrone128.yaml` points to a small committed subset of VisDrone with `96` train images, `16` validation images, and `16` test images for quick local checks and git-tracked examples.

### COCO128

`data/coco128.yaml` points to a lightweight COCO subset intended for quick local validation.

### COCO

`data/coco.yaml` defines the full COCO 2017 dataset. It is much larger than COCO128 and is not enabled for autodownload by default.

## Pushing to Tensorleap

After local validation passes, copy the visdrone128 datset to tensorleap accessable folder, 
update the leap_config.yaml and push the project with the current model:

```bash
leap push -m rtdetrv2_r18vd_120e_raw_outputs.onnx
```

If only code changes were made and the model asset did not change, prefer:

```bash
leap push
```

## Troubleshooting

- If visualizations look wrong, verify you are looking at the updated prediction visualizer path in `leap_binder/visualizers.py`.
- If dataset loading fails, check `dataset_name`, `dataset_autodownload`, and the selected dataset YAML.
- If platform validation fails while local validation passes, re-check model mapping and configured IO expectations.

## References

- [Tensorleap Docs](https://docs.tensorleap.ai/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [COCO Dataset](https://cocodataset.org/)
