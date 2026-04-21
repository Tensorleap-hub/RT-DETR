# S3 Per-Sample Data Loading Design

## Overview

Add per-sample S3 download to the RT-DETR Tensorleap integration. The dataset is defined by a COCO-format annotation JSON. Images are downloaded on demand when an encoder is called — only if the file does not already exist locally. The rest of the pipeline is unaffected.

---

## Architecture

```
leap_config.yaml (s3 block + annotation_file)
        │
        ▼
leap_config.py ──────────────────────────────────────────────┐
        │                                                     │
        ▼                                                     │
leap_binder/dataset.py (CocoDataset)                         │
  - loads annotation JSON at startup                         │
  - normalizes Windows backslashes in file_name              │
  - provides (image_path, annotations) by index              │
        │                                                     │
        ▼                                                     │
leap_binder/preprocess.py                                     │
  - input_encoder(idx)  → download image if missing → load   │
  - gt_encoder(idx)     → read annotations from memory       │
        │                                                     │
        ▼                                                     │
utils/aws_utils.py ◄─────────────────────────────────────────┘
  - _connect_to_s3()
  - download_file_if_missing(bucket, s3_key, local_path)
```

---

## Components

### `utils/aws_utils.py` (new)

- `_connect_to_s3() -> boto3.client`
  - Reads `AUTH_SECRET` env var (JSON string, same pattern as zeiss project)
  - Parses `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from it
  - Raises `ValueError` with clear message if env var is missing or malformed
  - Returns a configured boto3 S3 client

- `download_file_if_missing(bucket: str, s3_key: str, local_path: str) -> str`
  - Returns `local_path` immediately if file already exists (no-op on subsequent calls)
  - Creates parent directories as needed
  - Downloads via `s3_client.download_file(bucket, s3_key, local_path)`
  - Returns `local_path` after download
  - Raises on S3 errors (no silent fallback)

### `leap_binder/dataset.py` (new)

- `CocoDataset` class
  - Constructor: `__init__(annotation_path, dataset_root, s3_config=None)`
  - Loads COCO JSON from `annotation_path`
  - Normalizes `file_name` values: replaces `\\` with `/`
  - Builds `image_id → annotations` lookup dict from `annotations` list
  - Exposes:
    - `__len__()`: number of images
    - `__getitem__(idx)`: returns `(image_abs_path, image_meta, annotations_list)`
    - `get_image_s3_key(idx)`: returns `prefix + "/" + normalized_file_name`

### `leap_binder/preprocess.py` (modified)

- Replace `LoadImagesAndLabels` / `create_dataloader` with `CocoDataset`
- `preprocess_func_leap()`: instantiate `CocoDataset` per split using annotation JSON; return `PreprocessResponse(data=dataset, length=len(dataset))`
- `input_encoder(idx)`:
  1. `image_path, image_meta, _ = dataset[idx]`
  2. If `s3.enabled`: call `download_file_if_missing(bucket, s3_key, image_path)`
  3. Load image via OpenCV, resize to configured `image_size`, normalize to [0,1] with ImageNet mean/std
- `input_size_encoder(idx)`: return actual image `[height, width]` from `image_meta` (not hardcoded config value)
- `_padded_gt_for_sample(idx, preprocessing)` — **rewritten**:
  1. `_, image_meta, annotations = preprocessing.data[idx]`
  2. For each annotation: extract `category_id` and `bbox = [x, y, w, h]` (pixel coords)
  3. Convert to normalized `[cx, cy, nw, nh]`:
     - `cx = (x + w/2) / image_meta['width']`
     - `cy = (y + h/2) / image_meta['height']`
     - `nw = w / image_meta['width']`
     - `nh = h / image_meta['height']`
  4. Stack as `[category_id, cx, cy, nw, nh]`
  5. Pad to `max_num_of_objects` with `-1` rows; truncate if over limit
  6. Remove YOLOv5 rect-padding adjustments (not applicable to COCO)
- All downstream GT encoders (`gt_encoder`, `gt_boxes_encoder`, `gt_labels_encoder`, `gt_valid_mask_encoder`) call `_padded_gt_for_sample` — no changes to their signatures or slice logic

### `leap_config.yaml` (modified)

```yaml
# existing fields unchanged ...

annotation_file: "annotations_val.json"   # path relative to dataset_path

s3:
  enabled: false
  bucket_name: "my-bucket"
  prefix: "visdrone_ukraine"              # root prefix in bucket, no trailing slash
```

### `leap_config.py` (minor modification)

- Load and expose `annotation_file` and `s3` block from config
- No S3 logic here — purely config pass-through

---

## Data Flow: per-sample image load

```
input_encoder(idx)
  │
  ├── dataset[idx] → image_abs_path, s3_key
  │
  ├── [if s3.enabled]
  │     download_file_if_missing(bucket, s3_key, image_abs_path)
  │       └── already exists? → return immediately
  │           not exists?    → boto3 download → create dirs → save
  │
  └── load image from image_abs_path → preprocess → return tensor
```

## Data Flow: GT encoding

```
gt_encoder(idx)
  │
  ├── dataset[idx] → image_id, image meta (width, height)
  ├── lookup annotations by image_id (in-memory dict)
  └── convert + pad → return gt tensor
```

---

## S3 Key Derivation

Given:
- `file_name` in COCO JSON: `"attac-base\\images\\default\\35.png"`
- config `prefix`: `"visdrone_ukraine"`

S3 key: `"visdrone_ukraine/attac-base/images/default/35.png"`

Local path: `dataset_path / "attac-base/images/default/35.png"`

---

## Configuration: S3 vs Local

Set `s3.enabled: false` for local testing (default). Set `true` for S3 download. No other code changes needed.

---

## Auth

Credentials via `AUTH_SECRET` environment variable (JSON string):
```json
{"AWS_ACCESS_KEY_ID": "...", "AWS_SECRET_ACCESS_KEY": "..."}
```
No credentials in code or config files.

---

## Error Handling

- Missing `AUTH_SECRET`: raises `ValueError` at first download attempt
- S3 download failure: boto3 exception propagates (no silent fallback)
- Missing local file when S3 disabled: standard file-not-found error from image loader
