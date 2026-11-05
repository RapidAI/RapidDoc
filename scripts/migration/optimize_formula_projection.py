"""Create a separate, narrowly scoped ONNX candidate; never edit the source model."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx


SOURCE_SHA256 = '52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d'
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(destination):
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError('Source model differs from the investigated model')
    destination.mkdir(parents=True, exist_ok=False)
    model = onnx.load(SOURCE)
    graph = model.graph
    loop = next(node for node in graph.node if node.name == 'Loop.0')
    assert loop.op_type == 'Loop'
    body = next(attribute.g for attribute in loop.attribute if attribute.name == 'body')
    moves = [next(node for node in body.node if node.name == name)
             for name in ('MatMul.0', 'Add.20')]
    assert [(node.op_type, list(node.input), list(node.output)) for node in moves] == [
        ('MatMul', ['p2o.pd_op.transpose.0.0', 'linear_292.w_0'],
         ['p2o.sub_block.pd_op.matmul.0.0']),
        ('Add', ['p2o.sub_block.pd_op.matmul.0.0', 'linear_292.b_0'], ['Add.21'])]
    weights = [next(value for value in body.initializer if value.name == name)
               for name in ('linear_292.w_0', 'linear_292.b_0')]
    assert list(weights[0].dims) == [2048, 512] and list(weights[1].dims) == [512]
    weight_hashes = {value.name: hashlib.sha256(value.SerializeToString()).hexdigest()
                     for value in weights}
    # Prove that only outer-scope values and fixed initializers feed the move.
    available = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    available.update(value.name for value in weights)
    insertion = list(graph.node).index(loop)
    available.update(output for node in list(graph.node)[:insertion] for output in node.output)
    outer_names = {value.name for value in graph.initializer}
    outer_names.update(output for node in graph.node for output in node.output)
    assert not outer_names.intersection(value.name for value in weights)
    for node in moves:
        assert all(value in available for value in node.input)
        assert not outer_names.intersection(node.output)
        available.update(node.output)
    for index, node in enumerate(moves):
        graph.node.insert(insertion + index, node)
    graph.initializer.extend(weights)
    for node in moves:
        body.node.remove(node)
    for value in weights:
        body.initializer.remove(value)
    for value in graph.initializer:
        if value.name in weight_hashes:
            assert hashlib.sha256(value.SerializeToString()).hexdigest() == weight_hashes[value.name]
    target = destination / 'pp_formulanet_plus_m.hoisted.onnx'
    onnx.save_model(model, target)
    del model, graph, body, moves, weights
    onnx.checker.check_model(str(target))
    assert digest(SOURCE) == SOURCE_SHA256
    manifest = dict(source=str(SOURCE), source_sha256=SOURCE_SHA256, candidate=str(target),
                    candidate_sha256=digest(target), moved_nodes=['MatMul.0', 'Add.20'],
                    moved_initializer_sha256=weight_hashes, weights_modified=False,
                    source_modified=False, onnx_checker_passed=True)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
