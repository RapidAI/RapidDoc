"""Shared formula graph optimization for offline tools and in-memory CUDA loading.

MaxWithIndex requires the existing CUDA native library; keep retains standard ops.
The caller owns the ModelProto: optimize modifies it in place.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def subgraphs(node):
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            yield attr.g
        elif attr.type == onnx.AttributeProto.GRAPHS:
            yield from attr.graphs


def references(graph):
    """Include lexical references from nested graphs; conservative on collisions."""
    result = {name for node in graph.node for name in node.input if name}
    result.update(value.name for value in graph.output)
    for node in graph.node:
        for child in subgraphs(node):
            result.update(references(child))
    return result


def normalize_metadata(model):
    """Remove identical duplicate keys; reject ambiguous conflicting values."""
    values = {}
    removed = []
    for item in model.metadata_props:
        if item.key in values:
            if values[item.key] != item.value:
                raise ValueError(f"Conflicting ONNX metadata values for {item.key!r}")
            removed.append(item.key)
        else:
            values[item.key] = item.value
    if removed:
        helper.set_model_props(model, values)
    return removed


def optimize(model, argmax="maxindex"):
    if argmax not in ("topk", "maxindex", "keep"):
        raise ValueError("argmax must be topk, maxindex or keep")
    opset = next(v.version for v in model.opset_import if v.domain in ("", "ai.onnx"))
    if opset < 13 and argmax != "keep":
        raise ValueError("TopK/Squeeze rewrite requires ONNX opset >= 13")
    report = {"removed_loop_states": [], "argmax_rewrites": [], "skipped_argmax": [],
              "removed_duplicate_metadata": normalize_metadata(model)}
    all_names = set()

    def collect(graph):
        all_names.update(v.name for v in list(graph.input) + list(graph.output) + list(graph.initializer))
        for node in graph.node:
            all_names.update(node.input)
            all_names.update(node.output)
            for child in subgraphs(node):
                collect(child)

    collect(model.graph)

    def fresh(base):
        name = base
        suffix = 0
        while name in all_names:
            suffix += 1
            name = f"{base}_{suffix}"
        all_names.add(name)
        return name

    def visit(graph, inside_loop=False):
        for node in graph.node:
            for child in subgraphs(node):
                visit(child, inside_loop or node.op_type == "Loop")
            if node.op_type != "Loop" or node.domain not in ("", "ai.onnx"):
                continue
            body = next(a.g for a in node.attribute if a.name == "body")
            count = len(node.input) - 2
            if len(body.input) != count + 2 or len(body.output) != len(node.output) + 1:
                raise ValueError("Unexpected Loop input/output arity")
            body_refs, outer_refs = references(body), references(graph)
            dead = [i for i in range(count)
                    if body.input[i + 2].name not in body_refs and node.output[i] not in outer_refs]
            if dead:
                report["removed_loop_states"].append({"node": node.name, "before": count, "indices": dead})
                removed_outputs = []
                for i in dead:
                    value = onnx.ValueInfoProto()
                    value.CopyFrom(body.output[i + 1])
                    removed_outputs.append(value)
                for i in reversed(dead):
                    del node.input[i + 2]
                    del node.output[i]
                    del body.input[i + 2]
                    del body.output[i + 1]
                # Exporters may declare an intermediate tensor's type only in
                # Loop outputs. Keep that annotation for later CUDA rewrites.
                declared = {v.name for v in list(body.input) + list(body.output) + list(body.value_info)}
                for value in removed_outputs:
                    if value.name not in declared:
                        body.value_info.append(value)
                        declared.add(value.name)

        if not inside_loop or argmax == "keep":
            return
        types = {v.name: v.type.tensor_type.elem_type
                 for v in list(graph.input) + list(graph.value_info) + list(graph.output)
                 if v.type.HasField("tensor_type")}
        nodes = []
        for node in graph.node:
            if node.op_type != "ArgMax" or node.domain not in ("", "ai.onnx"):
                nodes.append(node)
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            reason = None
            if attrs.get("axis", 0) != -1:
                reason = "only explicit last-axis ArgMax is supported"
            elif attrs.get("select_last_index", 0):
                reason = "last-index tie handling is unsupported"
            elif argmax == "maxindex" and types.get(node.input[0]) != TensorProto.FLOAT:
                reason = "maxindex requires known FP32 input"
            if reason:
                report["skipped_argmax"].append({"node": node.name, "reason": reason})
                nodes.append(node)
                continue
            value = fresh(node.output[0] + "__max_value")
            keep = attrs.get("keepdims", 1)
            if argmax == "topk":
                k = fresh(node.output[0] + "__k")
                graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), k))
                index = node.output[0] if keep else fresh(node.output[0] + "__indices")
                nodes.append(helper.make_node("TopK", [node.input[0], k], [value, index],
                                              axis=-1, largest=1, sorted=1))
                if not keep:
                    axes = fresh(node.output[0] + "__axes")
                    graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), axes))
                    nodes.append(helper.make_node("Squeeze", [index, axes], list(node.output)))
            else:
                nodes.append(helper.make_node("MaxWithIndex", [node.input[0]], [value, node.output[0]],
                                              domain="onnxruntime_ops", keepdims=keep, largest=1))
            report["argmax_rewrites"].append({"output": node.output[0], "backend": argmax})
        del graph.node[:]
        graph.node.extend(nodes)

    visit(model.graph)
    if argmax == "maxindex" and report["argmax_rewrites"]:
        if not any(v.domain == "onnxruntime_ops" for v in model.opset_import):
            model.opset_import.append(helper.make_opsetid("onnxruntime_ops", 1))
    onnx.checker.check_model(model)
    return report


