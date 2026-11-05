"""Compute image-derived cross-attention K/V outside the token loop."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from build_formula_compression import SOURCE, SOURCE_SHA, digest, remove_internal_identities, store_half, graphs
from fuse_formula_attention import fuse, eliminate_dead_nodes, attribute


def hoist(model):
    outer = model.graph
    loop = next(node for node in outer.node if node.op_type == 'Loop')
    body = next(value.g for value in loop.attribute if value.name == 'body')
    branches = [node for node in body.node if node.op_type == 'If' and node.name in
                [f'If.{index}' for index in range(6)]]
    assert len(branches) == 6
    outer_names = {value.name for value in outer.initializer}
    outer_names.update(value for node in outer.node for value in node.output)
    additions, weights, removed_outputs, report = [], [], set(), []
    for branch in branches:
        otherwise = next(value.g for value in branch.attribute if value.name == 'else_branch')
        producer = {value: node for node in otherwise.node for value in node.output}
        initializers = {value.name: value for value in otherwise.initializer}
        for index, (branch_output, loop_output) in enumerate(zip(otherwise.output, branch.output)):
            transpose = producer[branch_output.name]
            assert transpose.op_type == 'Transpose' and attribute(transpose, 'perm') == [0, 2, 1, 3]
            reshape = producer[transpose.input[0]]
            assert reshape.op_type == 'Reshape'
            concat = producer[reshape.input[1]]
            assert concat.op_type == 'Concat' and attribute(concat, 'axis') == 0 and len(concat.input) == 4
            heads = numpy_helper.to_array(initializers[concat.input[2]])
            width = numpy_helper.to_array(initializers[concat.input[3]])
            sequence = numpy_helper.to_array(initializers[concat.input[1]])
            assert int(heads.item()) == 16 and int(width.item()) == 32 and int(sequence.item()) == -1
            add = producer[reshape.input[0]]
            assert add.op_type == 'Add'
            matmul = producer[add.input[0]]
            assert matmul.op_type == 'MatMul' and matmul.input[0] == 'Add.21'
            weight = initializers[matmul.input[1]]
            bias = initializers[add.input[1]]
            assert list(weight.dims) == [512, 512] and list(bias.dims) == [512]
            for value in [weight, bias]:
                assert value.name not in outer_names
                outer_names.add(value.name)
                weights.append(value)
            prefix = f'RapidDocCrossCache_{branch.name}_{index}'
            shape_name = prefix + '_shape'
            assert shape_name not in outer_names and loop_output not in outer_names
            # Image projection is [batch, image_sequence, 512]. Zero dimensions
            # copy its first two dimensions; no fixed batch or image length.
            weights.append(numpy_helper.from_array(np.array([0, 0, 16, 32], np.int64), shape_name))
            additions.extend([
                helper.make_node('MatMul', list(matmul.input), [prefix+'_projection'], name=prefix+'_matmul'),
                helper.make_node('Add', [prefix+'_projection', bias.name], [prefix+'_bias'], name=prefix+'_add'),
                helper.make_node('Reshape', [prefix+'_bias', shape_name], [prefix+'_heads'], name=prefix+'_reshape'),
                helper.make_node('Transpose', [prefix+'_heads'], [loop_output], perm=[0, 2, 1, 3], name=prefix+'_transpose')])
            removed_outputs.add(loop_output)
            report.append(dict(branch=branch.name, output=loop_output, image_input=matmul.input[0],
                               weight=weight.name, bias=bias.name))
    assert len(removed_outputs) == 12
    removed_positions = [index for index, value in enumerate(body.output) if value.name in removed_outputs]
    assert len(removed_positions) == 12 and min(removed_positions) >= 1
    discarded_outer = {loop.output[index-1] for index in removed_positions}
    assert not any(discarded_outer.intersection(node.input) for node in outer.node)
    assert not discarded_outer.intersection(value.name for value in outer.output)
    for index in sorted(removed_positions, reverse=True):
        del body.output[index]
        del body.input[index+1]
        del loop.input[index+1]
        del loop.output[index-1]
    nodes = [node for node in body.node if node.name not in {branch.name for branch in branches}]
    del body.node[:]
    body.node.extend(nodes)
    outer.initializer.extend(weights)
    insertion = list(outer.node).index(loop)
    for index, node in enumerate(additions):
        outer.node.insert(insertion+index, node)
    assert not eliminate_dead_nodes(outer)
    return report


def build(destination):
    assert digest(SOURCE) == SOURCE_SHA
    destination.mkdir(parents=True, exist_ok=False)
    model = onnx.load(SOURCE)
    remove_internal_identities(model.graph)
    fuse(model, 'all')
    report = hoist(model)
    weights = {value.name for graph in graphs(model.graph) for value in graph.initializer
               if value.data_type == TensorProto.FLOAT}
    store_half(model, weights, set())
    target = destination/'formula-cross-cache.onnx'
    onnx.save_model(model, target)
    del model
    onnx.checker.check_model(str(target))
    assert digest(SOURCE) == SOURCE_SHA
    result = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, candidate=str(target),
                  candidate_sha256=digest(target), hoisted_cross_cache=report,
                  removed_loop_carried_values=12, removed_if_nodes=6,
                  onnx_checker_passed=True, source_modified=False, installed_model_modified=False)
    (destination/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Cross-cache hoisted; 12 loop-carried values and 6 conditionals removed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
