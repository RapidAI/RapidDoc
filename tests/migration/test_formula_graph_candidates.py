"""Small semantic checks for experimental graph transformations, not model benchmarks."""
import sys
from pathlib import Path
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/migration'))
from build_formula_compression import remove_internal_identities
from formula_shape_constants import fold
from fuse_formula_attention import eliminate_dead_nodes, pack_linear_projections
from simplify_formula_shapes import simplify


def model(graph):
    result = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
    result.ir_version = 10
    return result


def run(value, inputs=None):
    onnx.checker.check_model(value)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    return ort.InferenceSession(value.SerializeToString(), options,
                               providers=['CPUExecutionProvider']).run(None, inputs or {})[0]


class GraphCandidates(unittest.TestCase):
    def test_shape_roundtrip_removal_keeps_dynamic_batch_and_length(self):
        graph = h.make_graph([
            h.make_node('Mul', ['x', 'scale'], ['scaled'], name='scale'),
            h.make_node('Shape', ['x'], ['shape'], name='shape'),
            h.make_node('Reshape', ['scaled', 'shape'], ['y'], name='roundtrip')], 'roundtrip',
            [h.make_tensor_value_info('x', T.FLOAT, [None, None, 8])],
            [h.make_tensor_value_info('y', T.FLOAT, [None, None, 8])],
            [nh.from_array(np.array([0.25], np.float32), 'scale')],
            value_info=[h.make_tensor_value_info('scaled', T.FLOAT, [None, None, 8])])
        value = model(graph)
        matches = simplify(value.graph)
        self.assertEqual(len(matches), 1)
        self.assertFalse(any(node.op_type == 'Reshape' for node in value.graph.node))
        for batch, length in [(1, 1), (2, 7)]:
            x = np.arange(batch * length * 8, dtype=np.float32).reshape(batch, length, 8)
            np.testing.assert_array_equal(run(value, {'x': x}), x * np.float32(0.25))

    def test_singleton_vector_does_not_preserve_scalar_rank(self):
        graph = h.make_graph([
            h.make_node('Mul', ['x', 'scale'], ['scaled'], name='scale'),
            h.make_node('Shape', ['x'], ['shape'], name='shape'),
            h.make_node('Reshape', ['scaled', 'shape'], ['y'], name='roundtrip')], 'scalar',
            [h.make_tensor_value_info('x', T.FLOAT, [])],
            [h.make_tensor_value_info('y', T.FLOAT, [])],
            [nh.from_array(np.array([0.25], np.float32), 'scale')],
            value_info=[h.make_tensor_value_info('scaled', T.FLOAT, [1])])
        value = model(graph)
        self.assertFalse(simplify(value.graph))
        self.assertTrue(any(node.op_type == 'Reshape' for node in value.graph.node))
        np.testing.assert_array_equal(run(value, {'x': np.array(4, np.float32)}), np.array(1, np.float32))

    def test_attention_cache_update_matches_explicit_concat(self):
        generator = np.random.default_rng(29)
        for batch, past_length, length in [(1, 0, 1), (2, 0, 1), (2, 11, 1), (1, 7, 3)]:
            query = generator.normal(size=(batch, length, 512)).astype(np.float32) / np.sqrt(np.float32(32))
            key = generator.normal(size=(batch, length, 512)).astype(np.float32)
            value = generator.normal(size=key.shape).astype(np.float32)
            past_key = generator.normal(size=(batch, 16, past_length, 32)).astype(np.float32)
            past_value = generator.normal(size=past_key.shape).astype(np.float32)
            bias = np.zeros((batch, 1, length, past_length + length), np.float32)
            for index in range(length):
                bias[:, :, index, past_length + index + 1:] = -10000
            feeds = dict(q=query, k=key, v=value, pk=past_key, pv=past_value, bias=bias)
            graph = h.make_graph([h.make_node('MultiHeadAttention',
                ['q', 'k', 'v', '', '', 'bias', 'pk', 'pv'], ['y', 'present_k', 'present_v'],
                domain='com.microsoft', num_heads=16, scale=1.0)], 'cache',
                [h.make_tensor_value_info(name, T.FLOAT, [None] * tensor.ndim)
                 for name, tensor in feeds.items()],
                [h.make_tensor_value_info('y', T.FLOAT, [None, None, 512]),
                 h.make_tensor_value_info('present_k', T.FLOAT, [None, 16, None, 32]),
                 h.make_tensor_value_info('present_v', T.FLOAT, [None, 16, None, 32])])
            value_model = model(graph)
            value_model.opset_import.append(h.make_opsetid('com.microsoft', 1))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.log_severity_level = 4
            result = ort.InferenceSession(value_model.SerializeToString(), options,
                                          providers=['CPUExecutionProvider']).run(None, feeds)
            expected_k = np.concatenate([past_key, key.reshape(batch, length, 16, 32).transpose(0, 2, 1, 3)], axis=2)
            expected_v = np.concatenate([past_value, value.reshape(batch, length, 16, 32).transpose(0, 2, 1, 3)], axis=2)
            np.testing.assert_array_equal(result[1], expected_k)
            np.testing.assert_array_equal(result[2], expected_v)
            scores = query.reshape(batch, length, 16, 32).transpose(0, 2, 1, 3) @ expected_k.transpose(0, 1, 3, 2)
            scores += bias
            probability = np.exp(scores - scores.max(-1, keepdims=True))
            probability /= probability.sum(-1, keepdims=True)
            expected = (probability @ expected_v).transpose(0, 2, 1, 3).reshape(batch, length, 512)
            np.testing.assert_allclose(result[0], expected, atol=2e-6, rtol=2e-5)

    def test_packed_projections_keep_weights_outputs_and_dynamic_lengths(self):
        generator = np.random.default_rng(13)
        weights = [generator.normal(size=(8, width)).astype(np.float32) for width in (4, 8, 3)]
        graph = h.make_graph([h.make_node('MatMul', ['x', f'w{i}'], [f'y{i}'], name=f'linear{i}')
                              for i in range(3)], 'packed',
                             [h.make_tensor_value_info('x', T.FLOAT, [None, None, 8])],
                             [h.make_tensor_value_info(f'y{i}', T.FLOAT, [None, None, weight.shape[1]])
                              for i, weight in enumerate(weights)],
                             [nh.from_array(weight, f'w{i}') for i, weight in enumerate(weights)])
        value = model(graph)
        report = pack_linear_projections(value)
        self.assertEqual(len(report), 1)
        self.assertEqual(sum(node.op_type == 'MatMul' for node in value.graph.node), 1)
        for batch, length in [(1, 1), (2, 3)]:
            x = generator.normal(size=(batch, length, 8)).astype(np.float32)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.log_severity_level = 4
            actual = ort.InferenceSession(value.SerializeToString(), options,
                                         providers=['CPUExecutionProvider']).run(None, {'x': x})
            for output, weight in zip(actual, weights):
                np.testing.assert_allclose(output, x @ weight, atol=1e-6, rtol=1e-5)

    def test_fused_attention_preserves_scaled_query_and_additive_mask(self):
        generator = np.random.default_rng(197)
        for batch, query_length, key_length, masked in [(1, 1, 1, False), (2, 1, 17, True),
                                                        (2, 3, 9, True)]:
            query = generator.normal(size=(batch, query_length, 512)).astype(np.float32) / np.sqrt(np.float32(32))
            key = generator.normal(size=(batch, 16, key_length, 32)).astype(np.float32)
            values = generator.normal(size=key.shape).astype(np.float32)
            inputs = ['q', 'k', 'v']
            feeds = dict(q=query, k=key, v=values)
            infos = [h.make_tensor_value_info('q', T.FLOAT, [None, None, 512]),
                     h.make_tensor_value_info('k', T.FLOAT, [None, 16, None, 32]),
                     h.make_tensor_value_info('v', T.FLOAT, [None, 16, None, 32])]
            scores = query.reshape(batch, query_length, 16, 32).transpose(0, 2, 1, 3) @ key.transpose(0, 1, 3, 2)
            if masked:
                bias = np.zeros((batch, 1, query_length, key_length), np.float32)
                bias[:, :, :, -1] = -10000
                scores += bias
                feeds['bias'] = bias
                infos.append(h.make_tensor_value_info('bias', T.FLOAT, [None, 1, None, None]))
                inputs += ['', '', 'bias']
            graph = h.make_graph([h.make_node('MultiHeadAttention', inputs, ['y'], domain='com.microsoft',
                                              num_heads=16, scale=1.0)], 'mha', infos,
                                 [h.make_tensor_value_info('y', T.FLOAT, [None, None, 512])])
            value = model(graph)
            value.opset_import.append(h.make_opsetid('com.microsoft', 1))
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            expected = (probabilities @ values).transpose(0, 2, 1, 3).reshape(batch, query_length, 512)
            np.testing.assert_allclose(run(value, feeds), expected, atol=2e-6, rtol=2e-5)

    def test_dead_node_cleanup_retains_implicit_loop_capture(self):
        body = h.make_graph([
            h.make_node('Identity', ['condition_in'], ['condition_out'], name='keep_condition'),
            h.make_node('Add', ['state', 'captured'], ['next_state'], name='step')], 'body',
            [h.make_tensor_value_info('iteration', T.INT64, []),
             h.make_tensor_value_info('condition_in', T.BOOL, []),
             h.make_tensor_value_info('state', T.INT64, [])],
            [h.make_tensor_value_info('condition_out', T.BOOL, []),
             h.make_tensor_value_info('next_state', T.INT64, [])])
        graph = h.make_graph([
            h.make_node('Add', ['three', 'three'], ['captured'], name='capture_producer'),
            h.make_node('Add', ['three', 'three'], ['unused'], name='dead'),
            h.make_node('Loop', ['trip', 'condition', 'three'], ['result'], body=body, name='loop')],
            'outer', [], [h.make_tensor_value_info('result', T.INT64, [])],
            [nh.from_array(np.array(2, np.int64), 'trip'), nh.from_array(np.array(True), 'condition'),
             nh.from_array(np.array(3, np.int64), 'three')])
        value = model(graph)
        self.assertFalse(eliminate_dead_nodes(value.graph))
        self.assertNotIn('dead', [node.name for node in value.graph.node])
        self.assertEqual(int(run(value)), 15)

    def test_partial_shape_retains_dynamic_batch(self):
        graph = h.make_graph([
            h.make_node('Shape', ['x'], ['shape']),
            h.make_node('Gather', ['shape', 'zero'], ['batch'], axis=0),
            h.make_node('Gather', ['shape', 'one'], ['width'], axis=0),
            h.make_node('Concat', ['batch', 'width'], ['result'], axis=0)], 'partial',
            [h.make_tensor_value_info('x', T.FLOAT, [None, 512])],
            [h.make_tensor_value_info('result', T.INT64, [2])],
            [nh.from_array(np.array([0], np.int64), 'zero'),
             nh.from_array(np.array([1], np.int64), 'one')],
            value_info=[h.make_tensor_value_info('width', T.INT64, [1])])
        value = model(graph)
        self.assertGreater(fold(value.graph), 0)
        self.assertTrue(any(node.op_type == 'Constant' and 'width' in node.output for node in value.graph.node))
        for batch in (1, 2, 7):
            np.testing.assert_array_equal(run(value, {'x': np.zeros((batch, 512), np.float32)}), [batch, 512])

    def test_loop_binding_does_not_capture_outer_alias(self):
        body = h.make_graph([
            h.make_node('Identity', ['condition_in'], ['condition_out']),
            h.make_node('Add', ['b', 'alias'], ['next_b'])], 'body',
            [h.make_tensor_value_info('iteration', T.INT64, []),
             h.make_tensor_value_info('condition_in', T.BOOL, []),
             h.make_tensor_value_info('b', T.INT64, [])],
            [h.make_tensor_value_info('condition_out', T.BOOL, []),
             h.make_tensor_value_info('next_b', T.INT64, [])])
        graph = h.make_graph([
            h.make_node('Identity', ['b'], ['alias']),
            h.make_node('Loop', ['trip', 'condition', 'b'], ['result'], body=body)], 'outer', [],
            [h.make_tensor_value_info('result', T.INT64, [])],
            [nh.from_array(np.array(2, np.int64), 'trip'),
             nh.from_array(np.array(True), 'condition'), nh.from_array(np.array(3, np.int64), 'b')])
        value = model(graph)
        self.assertEqual(int(run(value)), 9)
        remove_internal_identities(value.graph)
        self.assertEqual(int(run(value)), 9)
        fold(value.graph)
        self.assertEqual(int(run(value)), 9)


if __name__ == '__main__':
    unittest.main()
