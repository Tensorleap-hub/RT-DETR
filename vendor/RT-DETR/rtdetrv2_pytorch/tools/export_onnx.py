"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('RTDETR_IMPORT_MODE', 'minimal')

import torch
import torch.nn as nn

from src.core import YAMLConfig, yaml_utils


def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.concat_output = args.concat_output
            self.loss_outputs = args.loss_outputs

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
            if self.concat_output:
                # Export a single detection tensor in xyxy format:
                # [x1, y1, x2, y2, obj_conf, label]
                labels = labels.unsqueeze(-1).to(boxes.dtype)
                scores = scores.unsqueeze(-1)
                detections = torch.cat([boxes, scores, labels], dim=-1)
                if self.loss_outputs:
                    return detections, pred_logits, pred_boxes
                return detections
            if self.loss_outputs:
                return labels, boxes, scores, pred_logits, pred_boxes
            outputs = (labels, boxes, scores)
            return outputs

    model = Model()

    data = torch.rand(1, 3, args.input_size, args.input_size)
    size = torch.tensor([[args.input_size, args.input_size]])
    _ = model(data, size)

    if args.concat_output and args.loss_outputs:
        output_names = ['detections', 'pred_logits', 'pred_boxes']
    elif args.concat_output:
        output_names = ['detections']
    elif args.loss_outputs:
        output_names = ['labels', 'boxes', 'scores', 'pred_logits', 'pred_boxes']
    else:
        output_names = ['labels', 'boxes', 'scores']

    torch.onnx.export(
        model,
        (data, size),
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=output_names,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        dynamic = True 
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(args.output_file, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--concat-output', action='store_true', default=False,
                        help='Export single output [x1, y1, x2, y2, obj_conf, label]')
    parser.add_argument('--loss-outputs', action='store_true', default=False,
                        help='Also export raw pred_logits and pred_boxes for loss computation')
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config')

    args = parser.parse_args()

    main(args)
