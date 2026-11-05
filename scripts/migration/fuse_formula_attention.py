"""Fuse the verified formula decoder's attention subgraphs into CPU ORT MHA.

Keep projected/scaled Q, existing K/V caches, masks and output projections intact.
No training, quantization, vocabulary changes or document-specific conditions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from build_formula_compression import SOURCE, SOURCE_SHA, digest, graphs, remove_internal_identities, store_half


def eliminate_dead_nodes(graph):
    """Track live implicit subgraph captures as well as explicit node inputs."""
    captures = {}
    for node in graph.node:
        values = set()
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                values.update(eliminate_dead_nodes(attribute.g))
        captures[node.name] = values
    needed = {value.name for value in graph.output}
    retained = []
    for node in reversed(graph.node):
        if needed.intersection(node.output):
            retained.append(node)
            needed.update(value for value in node.input if value)
            needed.update(captures[node.name])
    del graph.node[:]
    graph.node.extend(reversed(retained))
    initializers = [value for value in graph.initializer if value.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    defined = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    defined.update(value for node in graph.node for value in node.output)
    infos = [value for value in graph.value_info if value.name in defined or value.name in needed]
    del graph.value_info[:]
    graph.value_info.extend(infos)
    return needed - defined


def attribute(node, name, default=None):
    return next((helper.get_attribute_value(value) for value in node.attribute if value.name == name), default)


def fuse(model, kind):
    body = next(attribute.g for node in model.graph.node if node.op_type == 'Loop'
                for attribute in node.attribute if attribute.name == 'body')
    producer = {value: node for node in body.node for value in node.output}
    users = {}
    for node in body.node:
        for value in node.input:
            users.setdefault(value, []).append(node)
    initializers = {value.name: value for graph in (model.graph, body) for value in graph.initializer}
    infos = {value.name: value for graph in (model.graph, body)
             for value in list(graph.input) + list(graph.value_info) + list(graph.output)}

    def op(value, expected):
        node = producer[value]
        if node.op_type != expected:
            raise ValueError(f'{value}: expected {expected}, got {node.op_type}')
        return node

    def next_op(value, expected):
        candidates = [node for node in users.get(value, []) if node.op_type == expected]
        if len(candidates) != 1:
            raise ValueError(f'{value}: expected one {expected} consumer')
        return candidates[0]

    def permutation(node, expected):
        assert attribute(node, 'perm') == expected, node.name

    replacements = {}
    matches = []
    for softmax in [node for node in body.node if node.op_type == 'Softmax']:
        assert attribute(softmax, 'axis', -1) == -1
        logits = producer[softmax.input[0]]
        mask = None
        if logits.op_type == 'MatMul':
            qk = logits
            category = 'cross'
        else:
            if kind == 'cross':
                continue
            add = op(op(softmax.input[0], 'Reshape').input[0], 'Add')
            options = []
            for index, value in enumerate(add.input):
                candidate = producer.get(value)
                if candidate is not None and candidate.op_type == 'Reshape':
                    inner = producer.get(candidate.input[0])
                    if inner is not None and inner.op_type == 'MatMul':
                        options.append((inner, add.input[1-index]))
            assert len(options) == 1, softmax.name
            qk, mask = options[0]
            category = 'self'
        qflat = op(qk.input[0], 'Reshape')
        qtranspose = op(qflat.input[0], 'Transpose')
        permutation(qtranspose, [0, 2, 1, 3])
        qheads = op(qtranspose.input[0], 'Reshape')
        shape = op(qheads.input[1], 'Concat')
        assert attribute(shape, 'axis') == 0 and len(shape.input) == 4
        dim = numpy_helper.to_array(initializers[shape.input[-1]])
        assert dim.size == 1 and int(dim.item()) > 0
        head_size = int(dim.item())
        query = qheads.input[0]
        query_type = infos[query].type.tensor_type
        assert query_type.elem_type == TensorProto.FLOAT and len(query_type.shape.dim) == 3
        head_count = numpy_helper.to_array(initializers[shape.input[-2]])
        assert head_count.size == 1 and int(head_count.item()) > 0
        heads = int(head_count.item())
        hidden = heads * head_size
        assert query_type.shape.dim[-1].dim_value in (0, hidden)
        kt = op(qk.input[1], 'Transpose')
        permutation(kt, [0, 2, 1])
        key = op(kt.input[0], 'Reshape').input[0]
        product = next_op(softmax.output[0], 'MatMul')
        assert product.input[0] == softmax.output[0]
        value = op(product.input[1], 'Reshape').input[0]
        for name in (key, value):
            assert len(infos[name].type.tensor_type.shape.dim) == 4, name
        restore_heads = next_op(product.output[0], 'Reshape')
        restore_transpose = next_op(restore_heads.output[0], 'Transpose')
        permutation(restore_transpose, [0, 2, 1, 3])
        context = next_op(restore_transpose.output[0], 'Reshape')
        context_type = infos[context.output[0]].type.tensor_type
        assert len(context_type.shape.dim) == 3 and context_type.shape.dim[-1].dim_value in (0, hidden)
        projection = next_op(context.output[0], 'MatMul')
        assert projection.input[0] == context.output[0]
        assert initializers[projection.input[1]].dims[0] == hidden
        inputs = [query, key, value]
        if mask is not None:
            inputs += ['', '', mask]
        # Q has already been scaled by the original graph; do not scale it twice.
        fused = helper.make_node('MultiHeadAttention', inputs, list(context.output), domain='com.microsoft',
                                 name='RapidDocMHA_' + softmax.name, num_heads=heads, scale=1.0)
        replacements[context.name] = fused
        matches.append(dict(softmax=softmax.name, kind=category, heads=heads, head_size=head_size,
                            query=query, key=key, value=value, attention_bias=mask,
                            context=context.output[0]))
    assert len(matches) == (6 if kind == 'cross' else 12), len(matches)
    before = len(body.node)
    rewritten = [replacements.get(node.name, node) for node in body.node]
    del body.node[:]
    body.node.extend(rewritten)
    if not any(value.domain == 'com.microsoft' for value in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    assert not eliminate_dead_nodes(model.graph), 'Unbound top-level values'
    remaining = {node.name for node in body.node}
    assert not any(row['softmax'] in remaining for row in matches), 'Unfused duplicate work remains live'
    return dict(matches=matches, body_nodes_before=before, body_nodes_after=len(body.node))


def pack_linear_projections(model):
    """Merge constant right-hand projections sharing the exact same input."""
    report = []
    for graph in graphs(model.graph):
        weights = {value.name: value for value in graph.initializer}
        groups = {}
        for node in graph.node:
            if node.op_type == 'MatMul' and node.input[1] in weights:
                weight = weights[node.input[1]]
                if weight.data_type == TensorProto.FLOAT and len(weight.dims) == 2:
                    groups.setdefault((node.input[0], weight.dims[0]), []).append(node)
        replacements, removed = {}, set()
        for (source, _), nodes in groups.items():
            if len(nodes) < 2:
                continue
            prefix = 'RapidDocPacked_' + nodes[0].name
            occupied = set(weights) | {value for node in graph.node for value in node.output}
            assert not {prefix + '_weight', prefix + '_sizes', prefix + '_output'}.intersection(occupied)
            combined = np.concatenate([numpy_helper.to_array(weights[node.input[1]]) for node in nodes], axis=1)
            sizes = np.array([weights[node.input[1]].dims[1] for node in nodes], dtype=np.int64)
            graph.initializer.extend([numpy_helper.from_array(combined, prefix + '_weight'),
                                      numpy_helper.from_array(sizes, prefix + '_sizes')])
            replacements[nodes[0].name] = [
                helper.make_node('MatMul', [source, prefix + '_weight'], [prefix + '_output'], name=prefix),
                helper.make_node('Split', [prefix + '_output', prefix + '_sizes'],
                                 [node.output[0] for node in nodes], axis=-1, name=prefix + '_split')]
            removed.update(node.name for node in nodes[1:])
            report.append(dict(input=source, projections=[node.name for node in nodes], widths=sizes.tolist()))
        rewritten = []
        for node in graph.node:
            if node.name not in removed:
                rewritten.extend(replacements.get(node.name, [node]))
        del graph.node[:]
        graph.node.extend(rewritten)
    assert not eliminate_dead_nodes(model.graph)
    return report


def fuse_cache_updates(model):
    """Let MHA produce present K/V instead of separate outer Concat nodes."""
    body = next(attribute.g for node in model.graph.node if node.op_type == 'Loop'
                for attribute in node.attribute if attribute.name == 'body')
    producer = {value: node for node in body.node for value in node.output}
    matches = []
    removed = set()
    for node in body.node:
        if node.op_type != 'MultiHeadAttention' or len(node.input) < 6 or not node.input[5]:
            continue
        caches = []
        for value in node.input[1:3]:
            while producer[value].op_type == 'Identity':
                value = producer[value].input[0]
            concat = producer[value]
            assert concat.op_type == 'Concat' and attribute(concat, 'axis') == 2
            assert len(concat.input) == 2
            transpose = producer[concat.input[1]]
            assert transpose.op_type == 'Transpose' and attribute(transpose, 'perm') == [0, 2, 1, 3]
            reshape = producer[transpose.input[0]]
            assert reshape.op_type == 'Reshape'
            caches.append(dict(past=concat.input[0], current=reshape.input[0], present=value,
                               concat=concat.name))
        node.input[1] = caches[0]['current']
        node.input[2] = caches[1]['current']
        node.input.extend([caches[0]['past'], caches[1]['past']])
        node.output.extend([caches[0]['present'], caches[1]['present']])
        removed.update(cache['concat'] for cache in caches)
        matches.append(dict(node=node.name, caches=caches))
    assert len(matches) == 6
    nodes = [node for node in body.node if node.name not in removed]
    del body.node[:]
    body.node.extend(nodes)
    assert not eliminate_dead_nodes(model.graph)
    # Repeated protobuf fields copy their children when rebuilt by cleanup.
    body = next(attribute.g for node in model.graph.node if node.op_type == 'Loop'
                for attribute in node.attribute if attribute.name == 'body')
    # New present-cache outputs may be consumed by nodes that used to precede MHA.
    # Sort explicit and implicit dependencies without guessing an insertion point.
    def captured(graph):
        used = {value.name for value in graph.output}
        defined = {value.name for value in graph.input} | {value.name for value in graph.initializer}
        for child in graph.node:
            used.update(value for value in child.input if value)
            defined.update(child.output)
            for item in child.attribute:
                if item.type == onnx.AttributeProto.GRAPH:
                    used.update(captured(item.g))
        return used - defined
    nodes = list(body.node)
    defined = {output for node in nodes for output in node.output}
    ready = {value for node in nodes for value in node.input if value not in defined}
    ready.update(value.name for value in body.input)
    ready.update(value.name for value in body.initializer)
    sorted_nodes = []
    while nodes:
        progress = False
        for node in nodes[:]:
            required = {value for value in node.input if value}
            for item in node.attribute:
                if item.type == onnx.AttributeProto.GRAPH:
                    required.update(captured(item.g) & defined)
            if required <= ready:
                sorted_nodes.append(node)
                ready.update(node.output)
                nodes.remove(node)
                progress = True
        if not progress:
            raise ValueError('Cache fusion introduced a dependency cycle')
    del body.node[:]
    body.node.extend(sorted_nodes)
    return matches


def build(output, kind, half, packed=False, cache=False):
    assert digest(SOURCE) == SOURCE_SHA
    output.mkdir(parents=True, exist_ok=False)
    model = onnx.load(SOURCE)
    remove_internal_identities(model.graph)
    result = fuse(model, kind)
    result['packed_projections'] = pack_linear_projections(model) if packed else []
    result['fused_cache_updates'] = fuse_cache_updates(model) if cache else []
    converted = []
    if half:
        weights = {value.name for graph in graphs(model.graph) for value in graph.initializer
                   if value.data_type == TensorProto.FLOAT}
        converted = store_half(model, weights, set())
    target = output / 'formula-attention.onnx'
    onnx.save_model(model, target)
    del model
    onnx.checker.check_model(str(target))
    assert digest(SOURCE) == SOURCE_SHA
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, candidate=str(target),
                    candidate_sha256=digest(target), **result, fp16_storage=half,
                    half_storage_weight_count=len(converted), onnx_checker_passed=True,
                    source_modified=False, installed_model_modified=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('MHA',kind, 'body nodes', result['body_nodes_before'], '->', result['body_nodes_after'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['cross', 'all'], default='all')
    parser.add_argument('--half-storage', action='store_true')
    parser.add_argument('--pack-projections', action='store_true')
    parser.add_argument('--fuse-cache', action='store_true')
    args = parser.parse_args()
    build(args.output.resolve(), args.kind, args.half_storage, args.pack_projections, args.fuse_cache)
