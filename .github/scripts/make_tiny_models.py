import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(opset, ir_version):
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((8, 3, 3, 3)).astype(np.float32), "conv_w")
    b = numpy_helper.from_array(rng.standard_normal((8,)).astype(np.float32), "conv_b")
    fw = numpy_helper.from_array(rng.standard_normal((8, 4)).astype(np.float32), "fc_w")
    fb = numpy_helper.from_array(rng.standard_normal((4,)).astype(np.float32), "fc_b")
    nodes = [
        helper.make_node("Conv", ["images", "conv_w", "conv_b"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["pool"]),
        helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["scores"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4]),
                               helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4])],
                              [w, b, fw, fb])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="tl-import-check-ci")
    if ir_version <= 8:
        onnx.checker.check_model(model)
    model.ir_version = ir_version
    return model


out = sys.argv[1] if len(sys.argv) > 1 else "."
onnx.save(build(16, 8), f"{out}/tiny_ok.onnx")
onnx.save(build(18, 9), f"{out}/tiny_new_format.onnx")
print("wrote tiny_ok.onnx (opset 16, IR 8) and tiny_new_format.onnx (opset 18, IR 9)")
