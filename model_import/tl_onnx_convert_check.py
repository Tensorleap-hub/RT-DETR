import argparse
import collections
import logging
import os
import platform
import sys
import time
import traceback

INSTALL_HINT = "python3.10 venv, then: pip install -r model_import/requirements.txt && pip install --no-deps onnx2kerastl==0.0.198 keras-data-format-converter==0.1.24"
EXPECTED_IR = 8
EXPECTED_OPSET = 16
EMPTY_INPUT_OK = ("Pad", "Resize", "Clip", "LSTM", "GRU")
EXPORT_STEPS = """Re-export the model before checking the conversion (README section 1):

  poetry run python export_onnx.py --config <config.yaml> --resume <checkpoint.pth> \\
      --output-file model.onnx --client-format --input-height 1088 --input-width 1920 --no-dynamic

  Exporting outside this repository with torch 2.9 or newer:
  torch.onnx.export(..., dynamo=False, opset_version=16).

Then rerun this script on the new file. Use --force to run the conversion on this file anyway."""


def section(title):
    print(f"\n=== {title} ===", flush=True)


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


def report_environment():
    section("Environment")
    print(f"python {sys.version.split()[0]}  {platform.platform()}")
    for p in ("tensorflow", "tensorflow-macos", "onnx", "onnxruntime", "onnx2kerastl", "keras-data-format-converter", "numpy", "protobuf"):
        print(f"{p:<30} {pkg_version(p)}")


def check_dependencies():
    missing = []
    for mod, pkg in (("tensorflow", "tensorflow"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("onnx2kerastl", "onnx2kerastl"), ("keras_data_format_converter", "keras-data-format-converter")):
        try:
            __import__(mod)
        except Exception as e:
            missing.append(f"{pkg} ({type(e).__name__}: {str(e)[:120]})")
    if missing:
        section("RESULT")
        print("NOT RUN: this Python environment is missing packages the check needs:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nActivate the venv created for this check, or set it up again: {INSTALL_HINT}")
        sys.exit(4)


def dims_of(vi):
    return [d.dim_value if d.dim_value else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]


def load_model(path):
    import onnx
    section("Model file")
    print(f"path: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    stub = onnx.load_model(path, load_external_data=False)
    ext = [t for t in stub.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if ext:
        print(f"{len(ext)} initializers are stored in an external data file ({ext[0].external_data[0].value}); it must sit next to the model")
    try:
        m = onnx.load_model(path)
    except Exception as e:
        print(f"FAILED to load the model the way Tensorleap does (onnx.load_model): {type(e).__name__}: {e}")
        section("RESULT")
        print("FAILED: the file could not be loaded. Export a single .onnx file (README section 1) and rerun.")
        sys.exit(1)
    g = m.graph
    print(f"ir_version: {m.ir_version}   opsets: {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}   producer: {m.producer_name} {m.producer_version}")
    for i in g.input:
        print(f"input  {i.name}: {onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)} {dims_of(i)}")
    for o in g.output:
        print(f"output {o.name}: {onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)} {dims_of(o)}")
    print(f"nodes: {len(g.node)}   initializers: {len(g.initializer)}")
    ops = collections.Counter(n.op_type for n in g.node)
    print("ops: " + ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])))
    return m


def format_findings(m):
    findings = []
    if m.ir_version > EXPECTED_IR:
        findings.append(f"IR version {m.ir_version} (expected {EXPECTED_IR} or lower)")
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=None)
    if opset is not None and opset > EXPECTED_OPSET:
        findings.append(f"opset {opset} (expected {EXPECTED_OPSET} or lower)")
    return findings


def check_format(m, force):
    section("Export format")
    findings = format_findings(m)
    if not findings:
        print(f"IR version {m.ir_version}, opset {EXPECTED_OPSET} or lower: OK")
        return
    print("This file does not match the format Tensorleap imports:")
    for f in findings:
        print(f"  - {f}")
    print()
    print(EXPORT_STEPS)
    if force:
        print("\ncontinuing because of --force")
        return
    section("RESULT")
    print("NOT CHECKED: the file must be re-exported first (see above).")
    sys.exit(2)


def precheck(m, force):
    section("Pre-check (graph structure)")
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    seen = set(inits) | {i.name for i in g.input}
    producer = {}
    blocking = []
    undefined = []
    for idx, n in enumerate(g.node):
        for k, inp in enumerate(n.input):
            if inp == "" and n.op_type in EMPTY_INPUT_OK:
                continue
            if inp not in seen:
                undefined.append((idx, n.op_type, n.name, k, inp))
        for o in n.output:
            seen.add(o)
            producer[o] = idx
    if undefined:
        blocking.append(f"{len(undefined)} node input(s) are not produced by any earlier node, initializer or graph input. The import stops at the first one.")
        for idx, op, name, k, inp in undefined[:10]:
            print(f"  node[{idx}] {op} '{name}' input#{k} = {inp!r} is not defined")
    else:
        print("all node inputs are defined before use: OK")
    missing_out = [o.name for o in g.output if o.name not in seen]
    if missing_out:
        blocking.append(f"graph output(s) {missing_out} are not produced by any node. The converted model would come out without them, and the import maps predictions by output index.")
    else:
        print(f"all {len(g.output)} graph outputs are produced: OK")

    def resolved_name(t):
        fl = t.ListFields()
        return fl[1][1] if len(fl) < 4 else fl[2][1]
    bad = [t.name for t in g.initializer if resolved_name(t) != t.name]
    if bad:
        blocking.append(f"{len(bad)} weight tensor(s) cannot be resolved by name: {bad[:10]}")
    else:
        print("all weight tensors resolve by name: OK")
    try:
        from onnx2kerastl.layers import AVAILABLE_CONVERTERS
        unsupported = sorted({n.op_type for n in g.node if n.op_type not in AVAILABLE_CONVERTERS})
        if unsupported:
            blocking.append(f"operator(s) not supported by the Tensorleap import: {', '.join(unsupported)}. Re-exporting with opset {EXPECTED_OPSET} usually removes them.")
        else:
            print("all operators are supported: OK")
    except Exception as e:
        print(f"could not read the supported operator table: {e}")
    splits = [n for n in g.node if n.op_type == "Split" and len(n.input) == 1 and not any(a.name == "split" for a in n.attribute) and len(n.output) != 2]
    if splits:
        blocking.append(f"{len(splits)} Split node(s) use the opset-18 'num_outputs' form, which is not supported. Exporting with opset {EXPECTED_OPSET} makes the split sizes explicit.")
    if blocking:
        print()
        print("Problems that will stop the import:")
        for b in blocking:
            print(f"  - {b}")
        if not force:
            section("RESULT")
            print("FAILED: fix the problems above (usually by re-exporting, README section 1) and rerun. Use --force to attempt the conversion anyway.")
            sys.exit(3)
        print("\ncontinuing because of --force")
    return producer, inits


def run_onnxruntime(path, m, skip_run):
    import numpy as np
    section("onnxruntime")
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"onnxruntime not available: {e}")
        return None, None
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"session created with onnxruntime {ort.__version__}: OK")
    except Exception as e:
        print(f"FAILED to load the model in onnxruntime {ort.__version__}: {type(e).__name__}: {str(e)[:700]}")
        return None, None
    if skip_run:
        return None, None
    feeds = {}
    rng = np.random.default_rng(0)
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
              "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_, "tensor(uint8)": np.uint8}.get(i.type, np.float32)
        feeds[i.name] = (rng.random(shape) > 0.5) if dt == np.bool_ else rng.random(shape).astype(dt)
    t0 = time.time()
    try:
        outs = sess.run(None, feeds)
        print(f"forward pass OK in {time.time() - t0:.1f}s: " + ", ".join(f"{o.name}{list(v.shape)}" for o, v in zip(sess.get_outputs(), outs)))
        return feeds, dict(zip([o.name for o in sess.get_outputs()], outs))
    except Exception as e:
        print(f"forward pass FAILED: {type(e).__name__}: {str(e)[:700]}")
        return feeds, None


def describe_input(name, m, producer, inits):
    import onnx
    from onnx import numpy_helper
    if name in inits:
        t = inits[name]
        desc = f"initializer {onnx.TensorProto.DataType.Name(t.data_type)}{list(t.dims)}"
        try:
            a = numpy_helper.to_array(t)
            if a.size <= 16:
                desc += f" = {a.tolist()}"
        except Exception:
            pass
        return desc
    if name in {i.name for i in m.graph.input}:
        return "graph input"
    if name in producer:
        n = m.graph.node[producer[name]]
        return f"output of node[{producer[name]}] {n.op_type} '{n.name}'"
    return "NOT DEFINED by any initializer, graph input or earlier node"


def failure_reason(e, node, m, producer, inits):
    msg = str(e)
    name = type(e).__name__
    if node is not None and "not in weights / model inputs / layers" in msg:
        missing = [inp for inp in node.input if inp and describe_input(inp, m, producer, inits).startswith("NOT DEFINED")]
        if missing:
            return f"input {missing[0]!r} is not defined by any initializer, graph input or earlier node"
        return "one of the node inputs could not be resolved"
    if name == "UnsupportedLayer" or "UnsupportedLayer" in msg:
        op = node.op_type if node is not None else msg.strip()
        return f"operator '{op}' is not supported by the Tensorleap import"
    lines = [l.strip() for l in msg.splitlines() if l.strip() and not l.startswith("Exception encountered when calling layer")]
    first = lines[0] if lines else name
    if "Dimension size" in msg or "shape" in msg.lower() or "reshape" in msg.lower():
        return f"shape mismatch: {first[:400]}"
    return f"{name}: {first[:400]}"


def check_output_count(k_model, m, stage):
    expected = [o.name for o in m.graph.output]
    if len(k_model.outputs) != len(expected):
        print(f"FAILED: after {stage} the model has {len(k_model.outputs)} output(s) but the ONNX file declares {len(expected)} {expected}. The Tensorleap import maps predictions by output index and stops here.")
        return False
    print(f"outputs after {stage}: {len(k_model.outputs)} (matches the ONNX file)")
    return True


def check_tensor_names(k_model):
    section("Tensor naming (Tensorleap graph wiring)")
    collisions = []
    checked = 0
    for layer in k_model.layers:
        for node in getattr(layer, "inbound_nodes", []):
            tensors = getattr(node, "keras_inputs", None)
            if tensors is None:
                try:
                    tensors = node.input_tensors
                except Exception:
                    continue
            tensors = tensors if isinstance(tensors, (list, tuple)) else [tensors]
            by_name = {}
            for t in tensors:
                try:
                    name = t.name
                except Exception:
                    continue
                hist = getattr(t, "_keras_history", None)
                ident = (hist.layer.name, hist.node_index, hist.tensor_index) if hist is not None else id(t)
                by_name.setdefault(name, set()).add(ident)
            checked += 1
            for name, idents in by_name.items():
                if len(idents) > 1:
                    collisions.append((layer.name, name, len(idents)))
    if not collisions:
        print(f"no layer receives two different tensors under one name ({checked} layer inputs checked): OK")
        return True
    print("FAILED: some layers receive two different tensors that carry the same name. The Tensorleap graph build wires layer arguments by tensor name, so one tensor would be fed into both slots. Locally the Keras model still runs; on the server this surfaces at the first inference (for example GatherNd 'does not index into param shape').")
    for layer_name, name, n in collisions[:10]:
        print(f"  layer '{layer_name}': {n} different tensors named {name!r}")
    print(f"  This is a converter-side issue, not a model issue. onnx2kerastl installed here: {pkg_version('onnx2kerastl')}. The known case (GatherElements) is fixed in 0.0.198, which Tensorleap servers 1.6.68 and newer run. If you already have 0.0.198, send us this output.")
    return False


def convert_with_tensorleap_flow(m, producer, inits, out_dir, stem, transform_io):
    from onnx2kerastl import onnx_to_keras
    section("Tensorleap conversion")
    ring = collections.deque(maxlen=80)
    warnings = collections.Counter()
    lg = logging.getLogger("onnx2keras")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    lg.handlers = []

    class Ring(logging.Handler):
        def emit(self, record):
            ring.append(f"{record.name}: {record.getMessage()}")
            if record.levelno >= logging.WARNING:
                warnings[f"{record.name}: {record.getMessage()}"] += 1
    debug_path = os.path.join(out_dir, f"{stem}.onnx2keras_debug.log")
    fh = logging.FileHandler(debug_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    lg.addHandler(Ring())
    lg.addHandler(fh)

    def print_warnings():
        if warnings:
            print("converter notes (the full ordered trace is in the debug log):")
            for msg, c in warnings.most_common(8):
                print(f"  x{c:<4} {msg[:160]}")
    input_all = [i.name for i in m.graph.input]
    input_initializer = [n.name for n in m.graph.initializer]
    input_names = [n for n in input_all if n not in input_initializer]
    print(f"debug log: {debug_path}")
    t0 = time.time()
    try:
        resp = onnx_to_keras(m, input_names=input_names, name_policy="attach_weights_name", allow_partial_compilation=False)
    except Exception as e:
        fh.flush()
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write("\n===== conversion failed, traceback =====\n")
            f.write(traceback.format_exc())
        node_name = next((l.split("node_name: ", 1)[1] for l in reversed(ring) if l.startswith("onnx2keras: node_name: ")), None)
        node = m.graph.node[producer[node_name]] if node_name in producer else None
        idx = producer.get(node_name)
        print()
        if node is not None:
            print(f"CONVERSION FAILED at node[{idx}] {node.op_type} '{node.name}'  ({idx / max(len(m.graph.node), 1):.0%} through the graph, after {time.time() - t0:.1f}s)")
        else:
            print(f"CONVERSION FAILED after {time.time() - t0:.1f}s")
        print(f"  reason: {failure_reason(e, node, m, producer, inits)}")
        if node is not None:
            import onnx
            attrs = {a.name: str(onnx.helper.get_attribute_value(a))[:60] for a in node.attribute}
            if attrs:
                print(f"  attributes: {attrs}")
            for k, inp in enumerate(node.input):
                print(f"  input#{k} {inp!r}: {describe_input(inp, m, producer, inits)}")
            print(f"  outputs: {list(node.output)}")
        print()
        print_warnings()
        print(f"The full converter trace and traceback are in: {debug_path}")
        return None
    print(f"conversion OK in {time.time() - t0:.1f}s")
    print_warnings()
    if getattr(resp, "error_info", None):
        print(f"partial-compilation info: {resp.error_info}")
    k_model = resp.converted_model
    if not check_output_count(k_model, m, "conversion"):
        return None
    section("Channels-last conversion")
    try:
        from keras_data_format_converter import convert_channels_first_to_last
        try:
            from onnx2kerastl.customonnxlayer import onnx_custom_layers
        except Exception:
            onnx_custom_layers = {}
        t0 = time.time()
        k_model = convert_channels_first_to_last(k_model, transform_io, onnx_custom_layers)
        print(f"OK in {time.time() - t0:.1f}s")
        if not check_output_count(k_model, m, "channels-last conversion"):
            return None
    except Exception as e:
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write("\n===== channels-last conversion failed, traceback =====\n")
            f.write(traceback.format_exc())
        print(f"FAILED: {type(e).__name__}: {str(e)[:400]}")
        print(f"Details are in: {debug_path}")
        return None
    if not check_tensor_names(k_model):
        return None
    section("Save and reload .h5")
    h5_path = os.path.join(out_dir, f"{stem}.tensorleap.h5")
    try:
        k_model.save(h5_path)
        import tensorflow as tf
        k_model = tf.keras.models.load_model(h5_path, custom_objects=onnx_custom_layers, compile=False)
        print(f"OK: {h5_path}")
    except Exception as e:
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write("\n===== h5 save/reload failed, traceback =====\n")
            f.write(traceback.format_exc())
        print(f"FAILED: {type(e).__name__}: {str(e)[:400]}")
        print(f"Details are in: {debug_path}")
        return None
    return k_model


def compare(k_model, feeds, ort_outs, m):
    import numpy as np
    section("Keras forward pass vs onnxruntime")
    if feeds is None:
        print("skipped (no onnxruntime reference)")
        return
    try:
        xs = [feeds[i.name] for i in m.graph.input if i.name in feeds]
        t0 = time.time()
        ys = k_model.predict(xs if len(xs) > 1 else xs[0], verbose=0)
        ys = ys if isinstance(ys, (list, tuple)) else [ys]
        print(f"keras forward pass OK in {time.time() - t0:.1f}s")
        if ort_outs is None:
            return
        for name, y in zip([o.name for o in m.graph.output], ys):
            ref = ort_outs[name]
            if ref.shape != y.shape:
                print(f"  {name}: shape mismatch onnx{list(ref.shape)} vs keras{list(y.shape)}")
                continue
            d = np.abs(ref.astype(np.float64) - y.astype(np.float64))
            print(f"  {name}: max|diff|={d.max():.3e}  mean|diff|={d.mean():.3e}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:400]}")


def main():
    ap = argparse.ArgumentParser(description=f"Run Tensorleap's ONNX import pipeline locally (format check, graph pre-check, onnxruntime, onnx2keras, channels-last, .h5) and report exactly where a model stops. Setup: {INSTALL_HINT}")
    ap.add_argument("onnx_path")
    ap.add_argument("--out-dir", help="where to write the .h5 and the debug log (default: next to the model)")
    ap.add_argument("--skip-run", action="store_true", help="do not run forward passes (faster)")
    ap.add_argument("--force", action="store_true", help="run the conversion even if the format check or pre-check reports problems")
    ap.add_argument("--transform-io", action="store_true", help="mirror the 'transform inputs' import option")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.WARNING)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    path = os.path.abspath(args.onnx_path)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    report_environment()
    check_dependencies()
    m = load_model(path)
    check_format(m, args.force)
    producer, inits = precheck(m, args.force)
    feeds, ort_outs = run_onnxruntime(path, m, args.skip_run)
    k_model = convert_with_tensorleap_flow(m, producer, inits, out_dir, stem, args.transform_io)
    if k_model is None:
        section("RESULT")
        print("FAILED: this model does not go through the Tensorleap import. Send us the console output above and the debug log.")
        sys.exit(1)
    if not args.skip_run:
        compare(k_model, feeds, ort_outs, m)
    section("RESULT")
    print("OK: this model goes through Tensorleap's import pipeline.")


if __name__ == "__main__":
    main()
