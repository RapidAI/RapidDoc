"""Remove provably redundant shape readbacks; no static batch/length assumptions."""
import argparse
import json
from pathlib import Path

import onnx
from onnx import helper

from build_formula_compression import digest, remove_internal_identities
from fuse_formula_attention import eliminate_dead_nodes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'build/formula-mha-half-20261029/formula-attention.onnx'
SHA = 'ee215a4db50b85bbfa3478ab81120bdb436beb3e86fd6f721b77effce53e871a'


def simplify(graph, inherited=None):
    """Track same-shape equivalence only, not equality of tensor contents."""
    local = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    local.update(output for node in graph.node for output in node.output)
    origin = {key: value for key, value in (inherited or {}).items() if key not in local}
    scalar = {value.name for value in graph.initializer if not value.dims}
    singleton_vector = {value.name for value in graph.initializer if list(value.dims) == [1]}
    ranks = {value.name: len(value.type.tensor_type.shape.dim)
             for value in list(graph.input) + list(graph.value_info) + list(graph.output)
             if value.type.tensor_type.HasField('shape')}
    producer = {value: node for node in graph.node for value in node.output}
    readbacks = {}
    matches = []
    rewritten = []
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                matches.extend(simplify(attribute.g, origin))
        root = lambda value: origin.get(value, value)
        if node.op_type in ('Identity', 'Cast', 'LayerNormalization', 'Relu', 'Erf'):
            if len(node.output) == 1:
                origin[node.output[0]] = root(node.input[0])
        elif node.op_type in ('Add', 'Sub', 'Mul', 'Div'):
            left, right = node.input
            for value, other in [(left, right), (right, left)]:
                if other in scalar or (other in singleton_vector and ranks.get(value, 0) >= 1):
                    origin[node.output[0]] = root(value)
                    break
            if root(left) == root(right):
                origin[node.output[0]] = root(left)
        if node.op_type == 'Shape' and len(node.attribute) == 0:
            identity = root(node.input[0])
            if identity in readbacks:
                matches.append(dict(node=node.name, action='reuse_shape', source=readbacks[identity]))
                node = helper.make_node('Identity', [readbacks[identity]], list(node.output), name=node.name)
            else:
                readbacks[identity] = node.output[0]
        elif node.op_type == 'Reshape':
            shape = producer.get(node.input[1])
            if shape is not None and shape.op_type == 'Shape' and not shape.attribute:
                if root(node.input[0]) == root(shape.input[0]):
                    matches.append(dict(node=node.name, action='remove_shape_roundtrip', source=node.input[0]))
                    origin[node.output[0]] = root(node.input[0])
                    node = helper.make_node('Identity', [node.input[0]], list(node.output), name=node.name)
        rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)
    return matches


def build(destination):
    assert digest(SOURCE) == SHA
    destination.mkdir(parents=True, exist_ok=False)
    model = onnx.load(SOURCE)
    matches = simplify(model.graph)
    remove_internal_identities(model.graph)
    assert not eliminate_dead_nodes(model.graph)
    target = destination / 'formula-shapes.onnx'
    onnx.save_model(model, target)
    del model
    onnx.checker.check_model(str(target))
    assert digest(SOURCE) == SHA
    manifest = dict(source=str(SOURCE), source_sha256=SHA, candidate=str(target), candidate_sha256=digest(target),
                    matches=matches, weights_modified=False, source_modified=False, onnx_checker_passed=True)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Removed or reused shape operations:', len(matches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
