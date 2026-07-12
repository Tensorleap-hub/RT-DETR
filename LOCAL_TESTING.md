# Local Testing

How to run the Tensorleap integration ([leap_integration.py](leap_integration.py)) end-to-end on your machine against a locally generated mock dataset — no MinIO/S3 or Tensorleap server required.

## Prerequisites

- Poetry environment for this repo (Python 3.10). All commands use `poetry run`.
- `code-loader >= 1.0.142` (already in the env).

## 1. Generate a mock dataset

The generator [scripts/create_mock_dataset.py](scripts/create_mock_dataset.py) writes a COCO-format dataset in the exact layout the integration expects:

```text
<root>/
  train/
    annotations_train.json      # file_name -> "pool_0000.jpg"
    images/pool_0000.jpg ...
  val/
    annotations_val.json
    images/...
  test/
    annotations_test.json
    images/...
```

Run it:

```bash
poetry run python scripts/create_mock_dataset.py
```

Default output root: `~/tensorleap/data/rheinmetall-mock`. Override with `--root <path>`.

### Setting the number of samples

Sample counts are the `SPLIT_COUNTS` constant at the top of the script:

```python
SPLIT_COUNTS = {"train": 400_000, "val": 50_000, "test": 50_000}   # 500K overall
```

Edit these numbers for larger or smaller runs.

### Large sample counts (shared image pool)

Generating one unique image per sample is impractical at scale — 500K unique
1920×1200 noise JPEGs would be **~693 GB**. Instead the generator writes a small
pool of physical images per split (`POOL_SIZE`, default `50`) and the COCO
annotations reference that pool round-robin. So:

- `preprocess` reports the full sample count (e.g. 500,000).
- On disk there are only `POOL_SIZE` images per split (~150 files, a few hundred MB).
- Each sample still gets its own randomized annotations.

Change the pool size with `--pool-size N` or the `POOL_SIZE` constant.

> Note: with 500K samples the per-split annotation JSON is large (~100+ MB) and
> is fully loaded into memory by `preprocess`, so the first run takes a few
> seconds longer.

## 2. Configure `leap_config.yaml`

For local testing the relevant keys in [leap_config.yaml](leap_config.yaml):

| Key | Local value | Notes |
|---|---|---|
| `dataset_path` | `~/tensorleap/data/rheinmetall-mock` | Where the generator wrote the data |
| `minio.enabled` | `false` | Disables S3/MinIO download; reads local files directly |
| `annotation_file` | `train/`, `val/`, `test/` entries | Relative to `dataset_path` |
| `model_path` | `client_format_structure.onnx` | ONNX model at repo root |
| `image_size` | `[1088, 1920]` | Input encoder resize target `[H, W]` |
| `gt_bbox_format` | `xywh_abs` | Matches the generator's COCO boxes |
| `pred_bbox_format` | `xyxy_abs` | Model output box format |
| `check_subset_index` | `0` | Which split `__main__` runs (0=train, 1=val, 2=test) |
| `check_sample_index` | `0` | Which sample index within that split |

When `minio.enabled: false`, no `AUTH_SECRET` or network access is needed.

## 3. Run the integration

```bash
poetry run python leap_integration.py
```

A successful run prints `Successful!` and a table of decorators, each marked ✅:

```text
Decorator Name                          | Added to integration
tensorleap_integration_test             | ✅
tensorleap_preprocess                   | ✅
tensorleap_input_encoder                | ✅
tensorleap_gt_encoder                   | ✅
tensorleap_load_model                   | ✅
...
```

`__main__` runs the sample selected by `check_subset_index` / `check_sample_index`.
To exercise every split and a few samples:

```bash
poetry run python -c "
from leap_binder import preprocess_func_leap
from leap_integration import check_integration
for s in preprocess_func_leap():
    for idx in [0, s.length - 1]:
        check_integration(idx, s)
    print(s.state, 'OK')
"
```

## 4. Structured validation (optional)

To validate the dataset side (`preprocess`, encoders, metadata) with `check_dataset()`:

```bash
poetry run python -c "
import os
from code_loader.leaploader import LeapLoader
r = LeapLoader(os.path.abspath('.'), 'leap_integration.py').check_dataset()
print('isValid:', r.is_valid)
print('generalError:', r.general_error)
"
```

`isValid: True` with no `generalError` means the dataset side is clean.
