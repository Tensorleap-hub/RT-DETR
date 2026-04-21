# Rheinmetall Client Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `rheinmetall` output format to the RT-DETR integration that matches the client's ONNX model: `boxes [B,300,4]`, `scores [B,300,3]`, `logits [B,300,3]`, non-square input, no `orig_target_sizes`.

**Architecture:** Add a `--client-format` export mode to `export_onnx.py` that bypasses the standard postprocessor and manually decodes boxes + computes softmax scores with background class dropped. Mirror the existing `_concat_scores` pattern in the binder — add parallel `_rheinmetall` variants of all visualizers, metrics, and losses, then wire them up in `leap_integration.py`. Verification is done by running `leap_integration.py` directly, not pytest.

**Tech Stack:** PyTorch, torchvision, onnxruntime, numpy, code_loader

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `export_onnx.py` | Modify | New `ClientFormatModel` class + `--client-format`, `--input-height`, `--input-width` args |
| `leap_config.py` | Modify | Add `"rheinmetall"` to `SUPPORTED_MODEL_OUTPUT_FORMATS` |
| `leap_config.yaml` | Modify | Commented rheinmetall config template |
| `leap_binder/common.py` | Modify | Add `format_rheinmetall_predictions()` and `image_scale_wh()` |
| `leap_binder/visualizers.py` | Modify | Add `bb_decoder_rheinmetall`, `pred_bb_decoder_rheinmetall` |
| `leap_binder/metrics.py` | Modify | Add `get_per_sample_metrics_rheinmetall`, `confusion_matrix_metric_rheinmetall` |
| `leap_binder/losses.py` | Modify | Add `detection_iou_loss_rheinmetall`, `detection_f1_loss_rheinmetall` |
| `leap_binder/__init__.py` | Modify | Export all new `_rheinmetall` symbols |
| `leap_integration.py` | Modify | Route `rheinmetall` format to new binder functions |

---

## Task 1: Export script — `--client-format` mode

**Files:**
- Modify: `export_onnx.py`

- [ ] **Step 1: Add `ClientFormatModel` and new CLI args to `export_onnx.py`**

Add `import torchvision` at the top if not already present.

Add after the existing `ExportModel` class (around line 76):

```python
class ClientFormatModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()

    def forward(self, images):
        outputs = self.model(images)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        h, w = images.shape[-2], images.shape[-1]
        boxes = torchvision.ops.box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes = boxes * scale

        logits = pred_logits[:, :, :-1]
        scores = torch.softmax(pred_logits, dim=-1)[:, :, :-1]

        return boxes, scores, logits
```

In `parse_args()`, add:
```python
parser.add_argument("--client-format", action="store_true", default=False,
                    help="Export in Rheinmetall client format: boxes, scores[B,N,C], logits[B,N,C]")
parser.add_argument("--input-height", type=int, default=None,
                    help="Input image height for non-square inputs")
parser.add_argument("--input-width", type=int, default=None,
                    help="Input image width for non-square inputs")
```

In `export()`, add a branch before the existing `ExportModel` instantiation:
```python
if args.client_format:
    h = args.input_height or args.input_size
    w = args.input_width or args.input_size
    model = ClientFormatModel(cfg)
    model.eval()
    data = torch.rand(args.batch_size, 3, h, w)
    _ = model(data)
    torch.onnx.export(
        model,
        (data,),
        args.output_file,
        input_names=["images"],
        output_names=["boxes", "scores", "logits"],
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    return
```

- [ ] **Step 2: Smoke-test the export**

```bash
poetry run python export_onnx.py \
  -c vendor/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -r rtdetrv2_r18vd_120e_raw_outputs.onnx \
  -o /tmp/client_format_test.onnx \
  --client-format --input-height 1088 --input-width 1920 \
  --no-dynamic
```

Verify outputs with a temp script:
```bash
poetry run python -c "
import onnx
m = onnx.load('/tmp/client_format_test.onnx')
for o in m.graph.output:
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, shape)
"
```

Expected:
```
boxes [1, 300, 4]
scores [1, 300, 79]
logits [1, 300, 79]
```
*(With Simon's 4-class model this will be `[1, 300, 3]`)*

- [ ] **Step 3: Commit**

```bash
git add export_onnx.py
git commit -m "feat: add client-format export mode for Rheinmetall"
```

---

## Task 2: Config + `common.py` foundations

**Files:**
- Modify: `leap_config.py`
- Modify: `leap_config.yaml`
- Modify: `leap_binder/common.py`

- [ ] **Step 1: Update `leap_config.py`** — add `"rheinmetall"` to supported formats

```python
SUPPORTED_MODEL_OUTPUT_FORMATS = {
    "rtdetr_raw",
    "detections",
    "detections_concat_scores",
    "rheinmetall",
}
```

- [ ] **Step 2: Add `image_scale_wh` and `format_rheinmetall_predictions` to `leap_binder/common.py`**

Add `image_scale_wh` after the imports:

```python
def image_scale_wh(image_size) -> np.ndarray:
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        h, w = image_size
    else:
        h = w = int(image_size)
    return np.array([w, h, w, h], dtype=np.float32)
```

Update `format_rtdetr_predictions` to accept an optional `_score_threshold` override:

```python
def format_rtdetr_predictions(
    labels: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    *,
    _score_threshold: float = None,
) -> np.ndarray:
    labels = np.asarray(labels).squeeze()
    boxes_xyxy = np.asarray(boxes_xyxy).squeeze()
    scores = np.asarray(scores).squeeze()

    if labels.ndim == 0:
        labels = np.array([labels], dtype=np.float32)
    if scores.ndim == 0:
        scores = np.array([scores], dtype=np.float32)
    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)

    score_threshold = _score_threshold if _score_threshold is not None else float(CONFIG.get("score_threshold", 0.3))
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
```

Add `format_rheinmetall_predictions` after `format_rtdetr_concat_predictions`:

```python
def format_rheinmetall_predictions(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    score_threshold: float = None,
) -> np.ndarray:
    boxes_xyxy = np.asarray(boxes_xyxy).squeeze()
    scores_per_class = np.asarray(scores_per_class).squeeze()

    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)
    if scores_per_class.ndim == 1:
        scores_per_class = scores_per_class.reshape(1, -1)

    scalar_scores = scores_per_class.max(axis=-1)
    labels = scores_per_class.argmax(axis=-1).astype(np.float32)

    return format_rtdetr_predictions(labels, boxes_xyxy, scalar_scores,
                                     _score_threshold=score_threshold)
```

- [ ] **Step 3: Update `leap_config.yaml`** — add commented rheinmetall template at the bottom:

```yaml
# Rheinmetall client format — uncomment to switch to Simon's model
# model_output_format: "rheinmetall"
# image_size: [1088, 1920]
# model_path: "simon_model.onnx"
# output_indices:
#   boxes: 0
#   scores: 1
#   logits: 2
```

- [ ] **Step 4: Verify with a temp script**

```bash
poetry run python -c "
import numpy as np
from leap_binder.common import image_scale_wh, format_rheinmetall_predictions

# image_scale_wh
print(image_scale_wh(640))          # [640. 640. 640. 640.]
print(image_scale_wh([1088, 1920])) # [1920. 1088. 1920. 1088.]

# format_rheinmetall_predictions
boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
scores = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
result = format_rheinmetall_predictions(boxes, scores, score_threshold=0.0)
print('label:', result[0, 0, 5])   # 1 (argmax)
print('score:', result[0, 0, 4])   # 0.7 (max)
"
```

- [ ] **Step 5: Commit**

```bash
git add leap_config.py leap_binder/common.py leap_config.yaml
git commit -m "feat: add rheinmetall format support and common helpers"
```

---

## Task 3: Visualizers — `_rheinmetall` variants

**Files:**
- Modify: `leap_binder/visualizers.py`

- [ ] **Step 1: Add visualizers**

Update the import at the top of `visualizers.py`:
```python
from .common import (
    format_rheinmetall_predictions,
    format_rtdetr_concat_predictions,
    format_rtdetr_predictions,
    label_names,
    prediction_rows,
)
```

Add after `pred_bb_decoder_concat_scores`:

```python
@tensorleap_custom_visualizer("bb_decoder_rheinmetall", LeapDataType.ImageWithBBox)
def bb_decoder_rheinmetall(
    image: np.ndarray,
    bb_gt: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    *,
    predictions: np.ndarray = None,
) -> LeapImageWithBBox:
    if predictions is None:
        predictions = format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    return bb_decoder_from_predictions(image, bb_gt, predictions)


@tensorleap_custom_visualizer("pred_bb_decoder_rheinmetall", LeapDataType.ImageWithBBox)
def pred_bb_decoder_rheinmetall(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    *,
    predictions: np.ndarray = None,
) -> LeapImageWithBBox:
    if predictions is None:
        predictions = format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    return pred_bb_decoder_from_predictions(image, predictions)
```

- [ ] **Step 2: Verify import works**

```bash
poetry run python -c "from leap_binder.visualizers import bb_decoder_rheinmetall, pred_bb_decoder_rheinmetall; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add leap_binder/visualizers.py
git commit -m "feat: add rheinmetall visualizers"
```

---

## Task 4: Metrics — `_rheinmetall` variants

**Files:**
- Modify: `leap_binder/metrics.py`

- [ ] **Step 1: Update import in `metrics.py`**

```python
from .common import (
    CONFIG,
    format_rheinmetall_predictions,
    format_rtdetr_concat_predictions,
    format_rtdetr_predictions,
    image_scale_wh,
    label_names,
    prediction_rows,
)
```

Replace `pred_boxes = pred[:, :4] / CONFIG["image_size"]` (appears in `get_per_sample_metrics_from_predictions` and `confusion_matrix_metric_from_predictions`) with:
```python
scale = image_scale_wh(CONFIG["image_size"])
pred_boxes = pred[:, :4] / scale
```

- [ ] **Step 2: Add `_rheinmetall` metric functions** at the end of `metrics.py`:

```python
@tensorleap_custom_metric(
    name="per_sample_metrics_rheinmetall",
    direction={
        "precision": MetricDirection.Upward,
        "recall": MetricDirection.Upward,
        "f1": MetricDirection.Upward,
        "FP": MetricDirection.Downward,
        "TP": MetricDirection.Upward,
        "FN": MetricDirection.Downward,
        "iou": MetricDirection.Upward,
        "accuracy": MetricDirection.Upward,
    },
)
def get_per_sample_metrics_rheinmetall(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
):
    y_preds = format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    return get_per_sample_metrics_from_predictions(y_preds, targets)


@tensorleap_custom_metric("Confusion Matrix Rheinmetall")
def confusion_matrix_metric_rheinmetall(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
):
    y_preds = format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    return confusion_matrix_metric_from_predictions(y_preds, targets)
```

- [ ] **Step 3: Verify import works**

```bash
poetry run python -c "from leap_binder.metrics import get_per_sample_metrics_rheinmetall, confusion_matrix_metric_rheinmetall; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add leap_binder/metrics.py
git commit -m "feat: add rheinmetall metrics, fix image_scale_wh normalization"
```

---

## Task 5: Losses — `_rheinmetall` variants

**Files:**
- Modify: `leap_binder/losses.py`

- [ ] **Step 1: Update import in `losses.py`**

```python
from .common import (
    COCO_CATEGORY_TO_LABEL,
    CONFIG,
    format_rheinmetall_predictions,
    format_rtdetr_concat_predictions,
    format_rtdetr_predictions,
    image_scale_wh,
    prediction_rows,
)
```

Replace `pred_boxes = pred[:, :4] / CONFIG["image_size"]` in `compute_detection_losses` with:
```python
scale = image_scale_wh(CONFIG["image_size"])
pred_boxes = pred[:, :4] / scale
```

- [ ] **Step 2: Add `_rheinmetall` loss functions** at the end of the detection loss section:

```python
@tensorleap_custom_loss("detection_iou_loss_rheinmetall")
def detection_iou_loss_rheinmetall(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(
        targets, y_preds=format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    )
    return losses["iou_loss"]


@tensorleap_custom_loss("detection_f1_loss_rheinmetall")
def detection_f1_loss_rheinmetall(
    boxes_xyxy: np.ndarray,
    scores_per_class: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    losses = compute_detection_losses(
        targets, y_preds=format_rheinmetall_predictions(boxes_xyxy, scores_per_class)
    )
    return losses["f1_loss"]
```

- [ ] **Step 3: Verify import works**

```bash
poetry run python -c "from leap_binder.losses import detection_iou_loss_rheinmetall, detection_f1_loss_rheinmetall; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add leap_binder/losses.py
git commit -m "feat: add rheinmetall losses, fix image_scale_wh in compute_detection_losses"
```

---

## Task 6: Wire up — `__init__.py` + `leap_integration.py`

**Files:**
- Modify: `leap_binder/__init__.py`
- Modify: `leap_integration.py`

- [ ] **Step 1: Update `leap_binder/__init__.py`**

Add new symbols to the existing imports and `__all__`:

```python
from .losses import (
    compute_detection_losses,
    compute_rtdetr_native_losses,
    detection_f1_loss,
    detection_f1_loss_concat_scores,
    detection_f1_loss_rheinmetall,
    detection_iou_loss,
    detection_iou_loss_concat_scores,
    detection_iou_loss_rheinmetall,
    rtdetr_loss_components_native,
    rtdetr_total_loss_native,
    yolov5_loss_factory,
    yolov5_new_loss,
)
from .metrics import (
    confusion_matrix_metric,
    confusion_matrix_metric_concat_scores,
    confusion_matrix_metric_rheinmetall,
    get_per_sample_metrics,
    get_per_sample_metrics_concat_scores,
    get_per_sample_metrics_rheinmetall,
)
from .visualizers import (
    bb_decoder,
    bb_decoder_concat_scores,
    bb_decoder_rheinmetall,
    image_visualizer,
    pred_bb_decoder,
    pred_bb_decoder_concat_scores,
    pred_bb_decoder_rheinmetall,
)
```

Add to `__all__`:
```python
"bb_decoder_rheinmetall",
"pred_bb_decoder_rheinmetall",
"confusion_matrix_metric_rheinmetall",
"get_per_sample_metrics_rheinmetall",
"detection_iou_loss_rheinmetall",
"detection_f1_loss_rheinmetall",
```

- [ ] **Step 2: Update `leap_integration.py`**

Add to the `from leap_binder import (...)` block:
```python
bb_decoder_rheinmetall,
confusion_matrix_metric_rheinmetall,
detection_f1_loss_rheinmetall,
detection_iou_loss_rheinmetall,
get_per_sample_metrics_rheinmetall,
pred_bb_decoder_rheinmetall,
```

Add a format flag after the existing ones (around line 42):
```python
MODEL_HAS_RHEINMETALL_FORMAT = MODEL_OUTPUT_FORMAT == "rheinmetall"
```

Extend the output indices block:
```python
if MODEL_HAS_RHEINMETALL_FORMAT:
    OUTPUT_INDICES["scores"] = _output_index("scores", 1)
    OUTPUT_INDICES["logits"] = _output_index("logits", 2)
```

Add a rheinmetall branch at the top of the selector block (before the existing `if MODEL_HAS_SEPARATE_SCORES`):
```python
if MODEL_HAS_RHEINMETALL_FORMAT:
    SELECTED_BB_DECODER = bb_decoder_rheinmetall
    SELECTED_PRED_BB_DECODER = pred_bb_decoder_rheinmetall
    SELECTED_PER_SAMPLE_METRIC = get_per_sample_metrics_rheinmetall
    SELECTED_CONFUSION_MATRIX_METRIC = confusion_matrix_metric_rheinmetall
    SELECTED_IOU_LOSS = detection_iou_loss_rheinmetall
    SELECTED_F1_LOSS = detection_f1_loss_rheinmetall
elif MODEL_HAS_SEPARATE_SCORES:
    SELECTED_BB_DECODER = bb_decoder
    ...
```

In `check_integration`, update the predictions block to branch on rheinmetall format:
```python
if MODEL_HAS_RHEINMETALL_FORMAT:
    boxes_output = predictions[OUTPUT_INDICES["boxes"]]
    scores_per_class = predictions[OUTPUT_INDICES["scores"]]
    vis_gt = SELECTED_BB_DECODER(image, gt, boxes_output, scores_per_class)
    vis_pred = SELECTED_PRED_BB_DECODER(image, boxes_output, scores_per_class)
    _ = SELECTED_PER_SAMPLE_METRIC(boxes_output, scores_per_class, gt)
    _ = SELECTED_CONFUSION_MATRIX_METRIC(boxes_output, scores_per_class, gt)
    _ = SELECTED_IOU_LOSS(boxes_output, scores_per_class, gt)
    _ = SELECTED_F1_LOSS(boxes_output, scores_per_class, gt)
else:
    # existing labels/boxes/scores block
    labels = predictions[OUTPUT_INDICES["labels"]]
    ...
```

Update the model `run` call to skip `orig_target_sizes` for rheinmetall:
```python
if MODEL_HAS_RHEINMETALL_FORMAT:
    predictions = model.run(None, {"images": image_for_model})
else:
    predictions = model.run(
        None,
        {"images": image_for_model, "orig_target_sizes": orig_sizes_for_model},
    )
```

- [ ] **Step 3: Verify existing format still works**

```bash
poetry run python leap_integration.py
```

Expected: runs without error (current config uses `rtdetr_raw`)

- [ ] **Step 4: Commit**

```bash
git add leap_binder/__init__.py leap_integration.py
git commit -m "feat: wire up rheinmetall format in leap_integration"
```
