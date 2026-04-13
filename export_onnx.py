import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parent
VENDOR_RTDETR_V1_ROOT = ROOT / "vendor" / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(VENDOR_RTDETR_V1_ROOT))

from src.core import YAMLConfig  # noqa: E402


def parse_cli_updates(nargs):
    if not nargs:
        return {}

    config = {}
    for entry in nargs:
        key, value = entry.split("=", 1)
        parsed_value = yaml.load(value, Loader=yaml.Loader)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parsed_value
    return config


def build_model(args):
    update_dict = parse_cli_updates(args.update)
    cfg = YAMLConfig(args.config, **update_dict)

    if not args.resume:
        raise ValueError("A checkpoint path is required for RT-DETR v1 export.")

    checkpoint = torch.load(args.resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    cfg.model.load_state_dict(state)
    return cfg


class ExportModel(nn.Module):
    def __init__(self, cfg, concat_output, loss_outputs) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.concat_output = concat_output
        self.loss_outputs = loss_outputs

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)

        if self.concat_output:
            scores = scores.unsqueeze(-1)
            boxes_with_score = torch.cat([boxes, scores], dim=-1)
            if self.loss_outputs:
                return labels, boxes_with_score, pred_logits, pred_boxes
            return labels, boxes_with_score

        if self.loss_outputs:
            return labels, boxes, scores, pred_logits, pred_boxes

        return labels, boxes, scores


def output_names(args):
    if args.concat_output and args.loss_outputs:
        return ["labels", "boxes", "pred_logits", "pred_boxes"]
    if args.concat_output:
        return ["labels", "boxes"]
    if args.loss_outputs:
        return ["labels", "boxes", "scores", "pred_logits", "pred_boxes"]
    return ["labels", "boxes", "scores"]


def dynamic_axes(args):
    axes = {
        "images": {0: "batch"},
        "orig_target_sizes": {0: "batch"},
    }
    for name in output_names(args):
        axes[name] = {0: "batch"}
    return axes


def export(args):
    cfg = build_model(args)
    model = ExportModel(cfg, concat_output=args.concat_output, loss_outputs=args.loss_outputs)
    model.eval()

    data = torch.rand(args.batch_size, 3, args.input_size, args.input_size)
    size = torch.full((args.batch_size, 2), args.input_size, dtype=torch.long)
    _ = model(data, size)

    export_kwargs = {}
    if args.dynamic:
        export_kwargs["dynamic_axes"] = dynamic_axes(args)

    torch.onnx.export(
        model,
        (data, size),
        args.output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=output_names(args),
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
        **export_kwargs,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
        onnx_model_simplify, check = onnxsim.simplify(
            args.output_file,
            input_shapes=input_shapes,
            dynamic_input_shape=True,
        )
        onnx.save(onnx_model_simplify, args.output_file)
        print(f"Simplify onnx model {check}...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--resume", "-r", type=str, required=True)
    parser.add_argument("--output-file", "--file-name", "-o", "-f", dest="output_file", type=str, default="model.onnx")
    parser.add_argument("--input-size", "-s", type=int, default=640)
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=False)
    parser.add_argument(
        "--concat-output",
        action="store_true",
        default=False,
        help="Export labels separately and boxes with score in the last channel [x1, y1, x2, y2, score].",
    )
    parser.add_argument(
        "--loss-outputs",
        action="store_true",
        default=False,
        help="Also export raw pred_logits and pred_boxes for loss computation",
    )
    parser.add_argument("--update", "-u", nargs="+", help="update yaml config")
    parser.set_defaults(dynamic=True)
    parser.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        help="Export symbolic batch axes.",
    )
    parser.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Export fixed batch axes using --batch-size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
