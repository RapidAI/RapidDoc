"""Conservative, in-memory replacement of supported standard ONNX operators."""
from dataclasses import dataclass, field
import copy
import onnx
from onnx import TensorProto, helper, numpy_helper
DOMAIN = "onnxruntime_ops"

@dataclass
class RewriteReport:
    conv: int = 0
    reductions: int = 0
    skipped: dict = field(default_factory=dict)
    enabled: bool = False
    reason: str = ""
    def skip(self, reason): self.skipped[reason] = self.skipped.get(reason, 0) + 1

def rewrite_model(model, *, cache_capacity=64, use_tf32=True, conv_api="classic", conv=True, reductions=True, validate=True):
    if not 0 <= cache_capacity <= 128: raise ValueError("cache_capacity must be between 0 and 128")
    if conv_api not in ("frontend", "classic"): raise ValueError("conv_api must be frontend or classic")
    m = copy.deepcopy(model)
    report = RewriteReport()
    try: typed = onnx.shape_inference.infer_shapes(m)
    except onnx.onnx_cpp2py_export.shape_inference.InferenceError: typed = m
    types = {v.name: (v.type.tensor_type.elem_type, len(v.type.tensor_type.shape.dim))
             for v in list(typed.graph.input) + list(typed.graph.value_info) + list(typed.graph.output)
             if v.type.HasField("tensor_type")}
    init = {v.name: v for v in m.graph.initializer}
    types.update({n: (v.data_type, len(v.dims)) for n, v in init.items()})
    attrs = lambda n: {v.name: helper.get_attribute_value(v) for v in n.attribute}
    def last_axis(name, axis):
        rank = types.get(name, (0, 0))[1]
        return rank > 0 and (axis + rank if axis < 0 else axis) == rank - 1
    replacement = {}; removed = set(); nodes = list(m.graph.node)
    if reductions:
        for i, a in enumerate(nodes):
            if a.domain not in ("", "ai.onnx"): continue
            aa = attrs(a)
            if a.op_type == "TopK" and len(a.input) == 2 and a.input[1] in init:
                if numpy_helper.to_array(init[a.input[1]]).tolist() != [1]: continue
                if types.get(a.input[0], (0, 0))[0] != TensorProto.FLOAT or not last_axis(a.input[0], aa.get("axis", -1)): continue
                replacement[i] = helper.make_node("MaxWithIndex", [a.input[0]], list(a.output), domain=DOMAIN, name=a.name,
                                                  keepdims=1, largest=aa.get("largest", 1))
                report.reductions += 1
            if a.op_type != "ArgMax" or aa.get("select_last_index", 0) != 0: continue
            if types.get(a.input[0], (0, 0))[0] != TensorProto.FLOAT or not last_axis(a.input[0], aa.get("axis", 0)): continue
            for j, b in enumerate(nodes):
                if j in removed or b.op_type != "ReduceMax" or b.domain not in ("", "ai.onnx") or b.input[0] != a.input[0]: continue
                ba = attrs(b); axes = ba.get("axes")
                if len(b.input) > 1 and b.input[1] in init: axes = numpy_helper.to_array(init[b.input[1]]).tolist()
                if axes is None or len(axes) != 1 or not last_axis(b.input[0], axes[0]): continue
                if ba.get("keepdims", 1) != aa.get("keepdims", 1): continue
                earliest = min(i, j)
                replacement[earliest] = helper.make_node("MaxWithIndex", [a.input[0]], [b.output[0], a.output[0]], domain=DOMAIN,
                                                         name=a.name, keepdims=aa.get("keepdims", 1), largest=1)
                removed.update((i, j)); report.reductions += 1; break
    for i, n in enumerate(nodes):
        if not conv or n.op_type != "Conv" or n.domain not in ("", "ai.onnx"): continue
        aa = attrs(n)
        if len(n.input) < 2 or types.get(n.input[1], (0, 0)) != (TensorProto.FLOAT, 4): report.skip("conv_type_or_rank"); continue
        auto = aa.get("auto_pad", b"NOTSET")
        if isinstance(auto,bytes):auto=auto.decode('ascii')
        if not auto:auto='NOTSET'
        if auto not in ('NOTSET','VALID','SAME_UPPER','SAME_LOWER'):report.skip('conv_auto_pad');continue
        pads = aa.get("pads", [0, 0, 0, 0]); strides = aa.get("strides", [1, 1]); dilations = aa.get("dilations", [1, 1])
        if len(pads) != 4 or pads[:2] != pads[2:] or min(pads) < 0: report.skip("conv_padding"); continue
        if len(strides) != 2 or len(dilations) != 2 or min(strides + dilations) <= 0: report.skip("conv_geometry"); continue
        replacement[i] = helper.make_node("CachedConv2D", list(n.input), list(n.output), domain=DOMAIN, name=n.name,
                                          pads=pads, strides=strides, dilations=dilations, group=aa.get("group", 1),
                                          cache_capacity=cache_capacity, use_tf32=int(use_tf32),auto_pad=auto,conv_api=conv_api)
        report.conv += 1
    out = [replacement[i] if i in replacement else n for i, n in enumerate(nodes) if i in replacement or i not in removed]
    del m.graph.node[:]; m.graph.node.extend(out)
    if report.conv or report.reductions:
        if not any(v.domain == DOMAIN for v in m.opset_import): m.opset_import.append(helper.make_opsetid(DOMAIN, 1))
        report.enabled = True
    # ORT optimized graphs can contain ORT-specific schemas in the ONNX domain
    # (e.g. SimplifiedLayerNormalization). ORT itself validates those graphs.
    if validate: onnx.checker.check_model(m)
    return m, report
