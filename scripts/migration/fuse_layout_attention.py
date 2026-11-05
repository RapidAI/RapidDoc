"""Fuse only standard layout attention, preserving deformable sampling and FP32 weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_formula_compression import digest, remove_internal_identities
from experiment_layout_compression import ROOT, SOURCES
from fuse_formula_attention import attribute, eliminate_dead_nodes


def fuse_standard_attention(model):
    graph = model.graph
    producer = {value: node for node in graph.node for value in node.output}
    users = {}
    for node in graph.node:
        for value in node.input:
            users.setdefault(value, []).append(node)
    weights = {value.name: value for value in graph.initializer}
    constants = {name: numpy_helper.to_array(value) for name, value in weights.items()}
    replacements, matches = {}, []

    def expect(value, kind):
        node = producer[value]
        assert node.op_type == kind, (node.name, kind)
        return node

    def one_user(value, kind):
        nodes = [node for node in users.get(value, []) if node.op_type == kind]
        assert len(nodes) == 1, (value, kind)
        return nodes[0]

    def projected(value):
        transpose = expect(value, 'Transpose')
        assert attribute(transpose, 'perm') == [0, 2, 1, 3]
        reshape = expect(transpose.input[0], 'Reshape')
        shape = constants[reshape.input[1]]
        assert shape.tolist() == [0, 0, 8, 32]
        return reshape.input[0]

    for softmax in [node for node in graph.node if node.op_type == 'Softmax']:
        value = softmax.input[0]
        upstream = producer.get(value)
        if upstream is not None and upstream.op_type == 'Add':
            zero = [name for name in upstream.input if name in constants and np.all(constants[name] == 0)]
            if len(zero) != 1:
                continue
            value = next(name for name in upstream.input if name != zero[0])
            upstream = producer.get(value)
        if upstream is None or upstream.op_type != 'Mul':
            continue
        scalar_names = [name for name in upstream.input if name in constants and constants[name].size == 1]
        if len(scalar_names) != 1:
            continue
        scale = float(constants[scalar_names[0]].item())
        value = next(name for name in upstream.input if name != scalar_names[0])
        qk = producer.get(value)
        if qk is None or qk.op_type != 'MatMul':
            continue
        assert attribute(softmax, 'axis', -1) == -1 and np.isfinite(scale) and scale > 0
        query = projected(qk.input[0])
        key_transpose = expect(qk.input[1], 'Transpose')
        assert attribute(key_transpose, 'perm') == [0, 1, 3, 2]
        key = projected(key_transpose.input[0])
        product = one_user(softmax.output[0], 'MatMul')
        assert product.input[0] == softmax.output[0]
        value = projected(product.input[1])
        restore = one_user(product.output[0], 'Transpose')
        assert attribute(restore, 'perm') == [0, 2, 1, 3]
        context = one_user(restore.output[0], 'Reshape')
        projection = one_user(context.output[0], 'MatMul')
        assert projection.input[0] == context.output[0] and weights[projection.input[1]].dims[0] == 256
        replacements[context.name] = helper.make_node('MultiHeadAttention', [query, key, value],
            list(context.output), domain='com.microsoft', num_heads=8, scale=scale,
            name='RapidDocLayoutMHA_'+softmax.name)
        matches.append(dict(softmax=softmax.name, query=query, key=key, value=value,
                            heads=8, head_width=32, scale=scale, output=context.output[0]))
    assert len(matches) > 0, 'No verified standard attention patterns'
    rewritten = [replacements.get(node.name, node) for node in graph.node]
    del graph.node[:]
    graph.node.extend(rewritten)
    if not any(value.domain == 'com.microsoft' for value in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    assert not eliminate_dead_nodes(graph)
    return matches


def build(kind, destination):
    name, sha = SOURCES[kind]
    source = ROOT.parent/'models'/name
    assert digest(source) == sha
    destination.mkdir(parents=True, exist_ok=False)
    model = onnx.load(source)
    remove_internal_identities(model.graph)
    matches = fuse_standard_attention(model)
    target = destination/('pp_doclayout'+kind+'.mha.onnx')
    onnx.save_model(model, target)
    del model
    onnx.checker.check_model(str(target))
    assert digest(source) == sha
    result = dict(kind=kind, source=str(source), source_sha256=sha, candidate=str(target),
                  candidate_sha256=digest(target), matches=matches, compute='FP32', weights_modified=False,
                  source_bytes=source.stat().st_size, candidate_bytes=target.stat().st_size,
                  onnx_checker_passed=True, source_modified=False, installed_model_modified=False)
    (destination/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(kind, 'standard attentions fused:', len(matches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['v2', 'v3'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.kind, args.output.resolve())
