# Model Export

## Recommended Command

```bash
poetry run python export_onnx.py \
  --config <path/to/config.yaml> \
  --resume <path/to/checkpoint.pth> \
  --output-file model.onnx \
  --client-format \
  --input-height 1088 \
  --input-width 1920 \
  --no-dynamic
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--config` | Path to the RT-DETR model YAML config (e.g. `configs/rtdetrv2_r50vd.yaml`) |
| `--resume` | Path to the trained `.pth` checkpoint |
| `--output-file` | Output ONNX file path |
| `--client-format` | Exports in integration-ready format: bakes in cxcywh→xyxy conversion and pixel-space scaling. Outputs `boxes [B,N,4]`, `scores [B,N,C]`, `logits [B,N,C]` |
| `--input-height` / `--input-width` | Input resolution — must match `image_size` in `leap_config.yaml` |
| `--no-dynamic` | Fixes the batch dimension (required for Tensorleap) |

## Output Format

With `--client-format` the model outputs three tensors:

- `boxes` — `[1, 300, 4]` bounding boxes in **xyxy pixel-space** coordinates
- `scores` — `[1, 300, C]` per-class softmax probabilities (background excluded)
- `logits` — `[1, 300, C]` raw logits (background excluded)

Set `boxes_in_cxcywh_format: false` in `leap_config.yaml` when using this export (default).

## Alternative: Raw Export

To export without the baked-in post-processing (raw cxcywh model output):

```bash
poetry run python export_onnx.py \
  --config <path/to/config.yaml> \
  --resume <path/to/checkpoint.pth> \
  --output-file model.onnx \
  --loss-outputs \
  --input-size 640 \
  --no-dynamic
```

Set `boxes_in_cxcywh_format: true` in `leap_config.yaml` for this format.
