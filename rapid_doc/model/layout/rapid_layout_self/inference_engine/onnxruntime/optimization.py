"""Standard ONNX-only compression of proven binary layout mask outputs."""
import copy

import onnx
from onnx import TensorProto, helper


def compact_binary_mask(model):
    """Return a copy with uint8 masks, or None for an unsupported graph.

    Only Cast/Reshape/Identity chains originating at a boolean comparison are
    accepted. Numeric masks/probabilities must never be truncated to uint8.
    Output positions, shapes, metadata and detection tensors remain unchanged.
    """
    if len(model.graph.output) != 3:
        return None
    output = model.graph.output[2]
    if output.type.tensor_type.elem_type != TensorProto.INT32 or len(output.type.tensor_type.shape.dim) != 3:
        return None
    producers = {name: node for node in model.graph.node for name in node.output}
    name, visited = output.name, set()
    while name not in visited:
        visited.add(name)
        node = producers.get(name)
        if node is None or node.domain not in ("", "ai.onnx"):
            return None
        if node.op_type in ("Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal"):
            break
        if node.op_type not in ("Cast", "Reshape", "Identity") or not node.input:
            return None
        if node.op_type == "Cast":
            dtype = next((a.i for a in node.attribute if a.name == "to"), None)
            if dtype not in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16,
                              TensorProto.BFLOAT16, TensorProto.INT32, TensorProto.INT64,
                              TensorProto.UINT8, TensorProto.BOOL):
                return None
        name = node.input[0]
    else:
        return None
    result = copy.deepcopy(model)
    names = {v.name for v in list(result.graph.input) + list(result.graph.output)
             + list(result.graph.value_info) + list(result.graph.initializer)}
    names.update(n for node in result.graph.node for n in list(node.input) + list(node.output))
    target = output.name + "__uint8"
    while target in names:
        target += "_"
    result.graph.node.append(helper.make_node("Cast", [output.name], [target], to=TensorProto.UINT8))
    result.graph.output[2].name = target
    result.graph.output[2].type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(result)
    return result
