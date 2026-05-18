# RT-DETR Tensorleap Integration

This repository contains a Tensorleap integration for object detection using an ONNX RT-DETR model. The integration entrypoint is `leap_integration.py`, with dataset preprocessing, metrics, visualizers, losses, and metadata implemented under `leap_binder/`.

All configuration lives in `leap_config.yaml`.

## Prerequisites

- Python `>=3.9,<3.11`
- Tensorleap CLI installed and authenticated if you plan to push to the platform

## Installation

```bash
pip install -r local_requirements.txt
```

Then run local commands directly:

```bash
python leap_integration.py
```

## Configuration

All fields are set in `leap_config.yaml`.

### Image size

```yaml
image_size: [1088, 1920]  # [height, width]
```

Height and width the model expects. Images are resized to this before inference.

### Model

```yaml
model_path: "model.onnx"
```

Path to the ONNX model file. Relative paths are resolved from the repository root.

The model is expected to expose at least two outputs:

- `output[0]`: bounding boxes, shape `[1, N, 4]`
- `output[1]`: class scores, shape `[1, N, num_classes]`

### Dataset

```yaml
dataset_path: "/path/to/dataset"

annotation_file:
  train: "train/annotations_train.json"
  val: "val/annotations_val.json"
```

`dataset_path` is the root folder of the dataset. `annotation_file` maps split names (`train`, `val`, `test`) to COCO-format JSON annotation paths relative to `dataset_path`. Only splits whose annotation files exist on disk are loaded.

Images are resolved under `<split_root>/images/` using the `file_name` field from each annotation entry. Windows-style backslashes in `file_name` are normalized automatically.

#### Label names

By default, class names are read from the `categories` field of the COCO annotation JSON. To override with a custom list, add:

```yaml
data_yaml_path: "data/labels.yaml"
```

The YAML file should contain a `names` or `pred_names` key with a list of class name strings. If the file is absent or neither key is present, the integration falls back to the COCO categories.

### S3

```yaml
s3:
  enabled: false
  bucket_name: ""
  prefix: ""
```

When `enabled: true`, annotation files and images are downloaded from S3 on demand before being read. Set `bucket_name` and `prefix` to match your bucket layout.

### Detection thresholds

```yaml
score_threshold: 0.3
max_detections: 300
max_num_of_objects: 500
```

- `score_threshold`: minimum confidence for a prediction to be kept in metrics and visualizers
- `max_detections`: maximum number of predictions kept per image after score filtering
- `max_num_of_objects`: maximum number of GT objects per image (padding target)

### Bounding box formats

The integration supports flexible bbox formats for both GT annotations and model outputs.

#### GT format

```yaml
# Valid values: "xywh_abs", "xywh_norm", "xyxy_abs", "xyxy_norm", "cxcywh_abs", "cxcywh_norm"
gt_bbox_format: "xywh_abs"
```

Describes the format of the `bbox` field in each COCO annotation entry:

| Value | Coordinate type | Scale |
| --- | --- | --- |
| `xywh_abs` | top-left x, y + width, height | absolute pixels (standard COCO) |
| `xywh_norm` | top-left x, y + width, height | normalized 0–1 |
| `xyxy_abs` | top-left and bottom-right corners | absolute pixels |
| `xyxy_norm` | top-left and bottom-right corners | normalized 0–1 |
| `cxcywh_abs` | center x, y + width, height | absolute pixels |
| `cxcywh_norm` | center x, y + width, height | normalized 0–1 |

#### Prediction format

```yaml
# Valid values: "xyxy_abs", "xyxy_norm", "cxcywh_abs", "cxcywh_norm"
pred_bbox_format: "xyxy_abs"
```

Describes the format of the boxes tensor produced by the ONNX model:

| Value | Coordinate type | Scale |
| --- | --- | --- |
| `xyxy_abs` | top-left and bottom-right corners | absolute pixels in model input resolution |
| `xyxy_norm` | top-left and bottom-right corners | normalized 0–1 |
| `cxcywh_abs` | center x, y + width, height | absolute pixels in model input resolution |
| `cxcywh_norm` | center x, y + width, height | normalized 0–1 |

### Loss weights

```yaml
loss:
  weight_dict:
    loss_vfl: 1.0
    loss_bbox: 5.0
    loss_giou: 2.0
  alpha: 0.75
  gamma: 2.0
  matcher:
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2.0
```

Weights and focal-loss parameters used by the RT-DETR detection losses.

### Local validation controls

```yaml
plot_visualizers: false
check_subset_index: 0
check_sample_index: 0
```

- `plot_visualizers`: when `true`, renders visualizer outputs to screen during `leap_integration.py`
- `check_subset_index`: which dataset split to use for the integration test (0 = first found split)
- `check_sample_index`: which sample within that split to test

## Local validation

Run the integration locally before pushing:

```bash
python leap_integration.py
```

This validates dataset loading, model inference, visualizers, metrics, losses, and metadata for a single sample.

## Project structure

| Path | Purpose |
| --- | --- |
| `leap_integration.py` | Tensorleap entrypoint and local integration test |
| `leap_config.yaml` | All project configuration |
| `leap_config.py` | Config loader and path resolution |
| `leap_binder/preprocess.py` | Preprocessing, input encoder, GT encoders |
| `leap_binder/visualizers.py` | Image and bounding box visualizers |
| `leap_binder/metrics.py` | Per-sample detection metrics and confusion matrix |
| `leap_binder/losses.py` | Detection IoU and F1 losses |
| `leap_binder/metadata.py` | Per-sample metadata (sharpness, bbox stats, class counts) |
| `leap_binder/common.py` | Shared utilities (bbox conversion, prediction formatting) |

## Secrets

When S3 access is required, provide AWS credentials to the platform as a secret using the Tensorleap CLI.

Create a JSON file with your credentials:

```json
{
  "AWS_ACCESS_KEY_ID": "your-access-key-id",
  "AWS_SECRET_ACCESS_KEY": "your-secret-access-key"
}
```

Then set the secret on the platform:

```bash
leap secrets set <secret-name> <path-to-credentials.json>
```

The secret name can be anything (e.g. `aws-credentials`). The platform will inject the keys as environment variables at runtime, which the integration picks up automatically when `s3.enabled: true`.

> **Note:** Never commit the credentials JSON file to version control. Add it to `.gitignore`.

## Pushing to Tensorleap

After local validation passes, push the project with the model:

```bash
leap push -m model.onnx
```

To push and start evaluate:

```bash
leap push -m model.onnx -n <version_name> --batch 1 --eval
```

If only code changed and the model asset is unchanged:

```bash
leap push
```

## Troubleshooting

- **Bounding boxes look wrong in size or position** — check `gt_bbox_format` and `pred_bbox_format`. Mismatched formats are the most common cause of displaced or oversized boxes.
- **No samples loaded** — verify `dataset_path` exists and that the annotation JSON paths under `annotation_file` resolve correctly.
- **Empty label names** — the COCO annotation JSON must have a non-empty `categories` field, or `data_yaml_path` must point to a valid label YAML.
- **S3 download failures** — confirm `s3.bucket_name`, `s3.prefix`, and AWS credentials are correct when `s3.enabled: true`.
- **Platform validation fails while local passes** — re-check that `dataset_path` and `model_path` are accessible from the platform environment.

## References

- [Tensorleap Docs](https://docs.tensorleap.ai/)
