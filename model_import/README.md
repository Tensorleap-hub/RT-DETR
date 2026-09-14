# Model import check

Tools to export the RT-DETR model in the format Tensorleap imports, and to run Tensorleap's ONNX-to-Keras import pipeline locally before uploading. If `tl_onnx_convert_check.py` ends with `RESULT: OK`, the same file will import on the server.

## 1. Export the model

Export from this repository's Poetry environment (torch 2.0.1), as described in [EXPORT.md](../EXPORT.md):

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

This produces a single `.onnx` file with IR version 8 and opset 16, the format Tensorleap imports.

If you export from another environment: torch 2.9 or newer defaults to the new exporter (`dynamo=True`), which writes IR version 10, opset 18 and a separate `.onnx.data` file. Pass `dynamo=False` and `opset_version=16` to `torch.onnx.export` to get the format above.

## 2. Run the import check

Environment constraints, matching the Tensorleap import environment:

| Constraint | Value |
|---|---|
| Python | 3.8, 3.9 or 3.10 (3.11 and newer are not supported by the pinned packages) |
| Platform | x86_64 Linux or Windows, or Apple silicon macOS |
| Pins | `onnx2kerastl==0.0.195`, `onnx==1.13.0`, `onnxruntime==1.17.3`, TensorFlow 2.12 (pulled in automatically) |
| Venv | Separate from the integration environment. The pins conflict with `local_requirements.txt` (onnx 1.17, tensorboard 2.20). |

Install (TensorFlow is a large download):

```bash
python3.10 -m venv tl-import-check
source tl-import-check/bin/activate        # Windows: tl-import-check\Scripts\activate
pip install -r model_import/requirements.txt
```

Run on the exact file you intend to upload:

```bash
python model_import/tl_onnx_convert_check.py path/to/model.onnx
```

The script runs in order: export format, graph pre-check, onnxruntime, conversion. If the format or the pre-check reports a problem it stops there with the fix (usually a re-export) and does not attempt the conversion, so failures are reported one cause at a time. Exit code 0 means the file is ready for `leap push`.

| Flag | Use |
|---|---|
| `--skip-run` | Skip the forward passes. Faster, still finds conversion failures. |
| `--force` | Attempt the conversion even if the format check or pre-check reported problems. |
| `--out-dir DIR` | Where to write the `.h5` and the debug log (default: next to the model). |
| `--transform-io` | Only if you import with the "transform inputs" option enabled in Tensorleap. |

### What it checks and why

| Step | Why it is there |
|---|---|
| Export format | IR version 8, opset 16 is the format Tensorleap imports. Anything else is sent back to section 1 before conversion is attempted. |
| `onnx.load_model` | The exact call the server uses. Catches a missing `.onnx.data` file or a corrupt file before anything else. |
| Inputs defined before use | Every node input must be a graph input, an initializer or the output of an earlier node. This is the rule that failed on the server; the script names the node. |
| Initializer names | Every weight tensor in the file must resolve by name. |
| Unsupported ops, opset-18 `Split` | Operators outside the supported set are listed up front instead of failing halfway. |
| onnxruntime load and run | Confirms the file itself is valid and runnable, with the onnxruntime version used on the server. |
| Conversion | The step that failed on the server, run with the server's exact settings. On failure the console shows one line with the cause, the node being converted and each of its inputs with its origin. The converter trace and traceback go to the debug log only. |
| Channels-last conversion, `.h5` save and reload | The remaining server-side steps of the import. |
| Keras vs onnxruntime outputs | Numerical sanity check of the converted model. |

Outputs written next to the model (or to `--out-dir`): `<model>.onnx2keras_debug.log` (full converter trace and any traceback, text only) and `<model>.tensorleap.h5`.

## 3. If it fails

Send us:

1. The full console output and the `<model>.onnx2keras_debug.log` file. Both are text only and contain no weights.
2. The `torch.onnx.export(...)` call you used and the output of `python -c "import torch; print(torch.__version__)"`.

If we need the graph itself, this produces a copy with the float weights replaced by noise and everything else intact. Run the check on the copy first to confirm it fails the same way:

```python
import onnx, numpy as np
from onnx import numpy_helper
m = onnx.load("model.onnx")
for t in m.graph.initializer:
    if t.data_type == onnx.TensorProto.FLOAT:
        a = numpy_helper.to_array(t)
        r = np.random.randn(*a.shape).astype(np.float32)
        if t.name.endswith("running_var"):
            r = np.abs(r) + 1e-3
        t.raw_data = r.tobytes()
        del t.float_data[:]
onnx.save(m, "model_anonymized.onnx")
```
