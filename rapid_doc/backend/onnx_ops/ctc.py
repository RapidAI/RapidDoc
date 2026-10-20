"""Explicit CTC output reduction; unlike create_session this changes outputs."""
import copy
from pathlib import Path
import onnx
from onnx import helper, TensorProto

def compact_ctc_model(source):
    """Return an in-memory model with indices/scores instead of [N,T,C].

    Input must already output probabilities (no implicit softmax is added).
    Callers must adapt CTC decoding to the two outputs, in that order.
    """
    if isinstance(source, onnx.ModelProto): model = copy.deepcopy(source)
    elif isinstance(source, (str, Path)): model = onnx.load(str(source))
    else: model = onnx.load_model_from_string(source)
    if len(model.graph.output) != 1:
        raise ValueError("CTC compaction requires one probability output")
    output = model.graph.output[0]
    tensor = output.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 3:
        raise ValueError("CTC compaction requires float32 probabilities [N,T,C]")
    shape = [d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in tensor.shape.dim][:2]
    names = {s for n in model.graph.node for s in list(n.input) + list(n.output)}
    names.update(i.name for i in model.graph.initializer)
    def fresh(base):
        result = base
        while result in names: result += '_'
        names.add(result)
        return result
    indices, scores, axes = (fresh(n) for n in ('ortops_ctc_indices','ortops_ctc_scores','ortops_ctc_axes'))
    opset = next(o.version for o in model.opset_import if o.domain in ('','ai.onnx'))
    model.graph.node.append(helper.make_node('ArgMax',[output.name],[indices],axis=2,keepdims=0))
    if opset >= 18:
        model.graph.initializer.append(helper.make_tensor(axes,TensorProto.INT64,[1],[2]))
        node = helper.make_node('ReduceMax',[output.name,axes],[scores],keepdims=0)
    else:
        node = helper.make_node('ReduceMax',[output.name],[scores],axes=[2],keepdims=0)
    model.graph.node.append(node)
    del model.graph.output[:]
    model.graph.output.extend([helper.make_tensor_value_info(indices,TensorProto.INT64,shape),helper.make_tensor_value_info(scores,TensorProto.FLOAT,shape)])
    onnx.external_data_helper.convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    return model

def create_ctc_session(model, **kwargs):
    """Create a session outputting [indices, scores]; explicit opt-in only."""
    from .session import create_session
    return create_session(compact_ctc_model(model), **kwargs)
