"""Build isolated storage/quantization experiments from the verified hoisted graph."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'build/formula-projection-candidate-20261029/pp_formulanet_plus_m.hoisted.onnx'
SOURCE_SHA = 'a9d91509f48933088f9413ff22db61c3d25e0d588ede466a089c9845600056a2'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from graphs(attribute.g)


def remove_internal_identities(graph, inherited=None):
    """Resolve aliases lexically, retaining all graph-output identities."""
    local = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    local.update(value for node in graph.node for value in node.output)
    aliases = {key: value for key, value in (inherited or {}).items() if key not in local}
    outputs = {value.name for value in graph.output}
    # An alias target must not become captured by a descendant's local binding.
    descendants = list(graphs(graph))[1:]
    shadowed = {value.name for child in descendants for value in child.input}
    shadowed.update(value.name for child in descendants for value in child.initializer)
    shadowed.update(value for child in descendants for node in child.node for value in node.output)
    descendant_outputs = {value.name for child in descendants for value in child.output}
    nodes, removed = [], set()
    for node in graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = aliases.get(value, value)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                remove_internal_identities(attribute.g, aliases)
        if (node.op_type == 'Identity' and node.output[0] not in outputs
                and node.output[0] not in descendant_outputs and node.input[0] not in shadowed):
            aliases[node.output[0]] = node.input[0]
            removed.add(node.output[0])
        else:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    infos = [value for value in graph.value_info if value.name not in removed]
    del graph.value_info[:]
    graph.value_info.extend(infos)


def store_half(model, original_weights, exclusions):
    converted = []
    for graph in graphs(model.graph):
        casts = []
        input_names = {value.name for value in graph.input}
        occupied = {value.name for value in graph.initializer}
        occupied.update(value for node in graph.node for value in node.output)
        for value in graph.initializer:
            if (value.name not in original_weights or value.name in exclusions or value.name in input_names
                    or value.data_type != TensorProto.FLOAT or np.prod(value.dims) < 128):
                continue
            name = value.name
            stored = name + '__storage_fp16'
            assert stored not in occupied
            array = numpy_helper.to_array(value)
            half = array.astype(np.float16)
            if not np.all(np.isfinite(half)):
                raise ValueError(f'FP16 storage overflow: {name}')
            value.CopyFrom(numpy_helper.from_array(half, stored))
            casts.append(helper.make_node('Cast', [stored], [name], to=TensorProto.FLOAT,
                                          name=name + '__restore_fp32'))
            converted.append(dict(name=name, elements=int(array.size),
                                  max_absolute_weight_error=float(np.max(np.abs(array - half.astype(np.float32))))))
        nodes = casts + list(graph.node)
        del graph.node[:]
        graph.node.extend(nodes)
    return converted


def build(mode, destination, exclusions, shape_fold=False):
    assert digest(SOURCE) == SOURCE_SHA
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / f'formula-{mode}.onnx'
    model = onnx.load(SOURCE)
    original_weights = {value.name for graph in graphs(model.graph) for value in graph.initializer
                        if value.data_type == TensorProto.FLOAT}
    remove_internal_identities(model.graph)
    folded = 0
    if shape_fold:
        from formula_shape_constants import fold
        folded = fold(model.graph)
    if mode in ('int8', 'hybrid'):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        logging.getLogger().setLevel(logging.ERROR)
        quantize_dynamic(model, target, op_types_to_quantize=['MatMul', 'Gather'], per_channel=True,
                         weight_type=QuantType.QInt8, nodes_to_exclude=exclusions,
                         extra_options={'EnableSubgraph': True, 'MatMulConstBOnly': True})
        del model
        if mode == 'hybrid':
            model = onnx.load(target)
    converted = []
    if mode in ('fp16-storage', 'hybrid'):
        converted = store_half(model, original_weights, set(exclusions))
        onnx.save_model(model, target)
        del model
    elif mode == 'fp32-clean':
        onnx.save_model(model, target)
        del model
    onnx.checker.check_model(str(target))
    assert digest(SOURCE) == SOURCE_SHA
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, candidate=str(target),
                    candidate_sha256=digest(target), mode=mode, source_bytes=SOURCE.stat().st_size,
                    candidate_bytes=target.stat().st_size, half_storage_weights=converted,
                    exclusions=exclusions, source_modified=False, installed_model_modified=False,
                    integer_shape_constants_folded=folded,
                    quantization_is_lossy=mode != 'fp32-clean', onnx_checker_passed=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(mode, 'bytes', target.stat().st_size, 'half weights', len(converted), 'integer constants', folded, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fp32-clean', 'fp16-storage', 'int8', 'hybrid'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude', action='append', default=[])
    parser.add_argument('--shape-fold', action='store_true')
    args = parser.parse_args()
    build(args.mode, args.output.resolve(), args.exclude, args.shape_fold)
