"""Check layout attention replacement numerics and conservative mask matching."""
from pathlib import Path
import sys
import unittest

import numpy as np
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/migration'))
from fuse_layout_attention import fuse_standard_attention


def graph_model(mask):
    nodes = []
    for name in ['q', 'k', 'v']:
        nodes.extend([h.make_node('Reshape', [name, 'heads'], [name+'h'], name=name+'_reshape'),
                      h.make_node('Transpose', [name+'h'], [name+'t'], perm=[0, 2, 1, 3], name=name+'_transpose')])
    nodes.extend([
        h.make_node('Transpose', ['kt'], ['kt2'], perm=[0, 1, 3, 2], name='key_t'),
        h.make_node('MatMul', ['qt', 'kt2'], ['qk'], name='qk'),
        h.make_node('Mul', ['qk', 'scale'], ['scaled'], name='scale'),
        h.make_node('Add', ['scaled', 'mask'], ['masked'], name='mask'),
        h.make_node('Softmax', ['masked'], ['probability'], axis=-1, name='softmax'),
        h.make_node('MatMul', ['probability', 'vt'], ['context'], name='context'),
        h.make_node('Transpose', ['context'], ['context_t'], perm=[0, 2, 1, 3], name='restore_t'),
        h.make_node('Reshape', ['context_t', 'restore'], ['flat'], name='flatten'),
        h.make_node('MatMul', ['flat', 'projection'], ['y'], name='projection')])
    graph = h.make_graph(nodes, 'layout_attention',
        [h.make_tensor_value_info(name, T.FLOAT, [None, None, 256]) for name in ['q', 'k', 'v']],
        [h.make_tensor_value_info('y', T.FLOAT, [None, None, 256])],
        [nh.from_array(np.array([0, 0, 8, 32], np.int64), 'heads'),
         nh.from_array(np.array([0, 0, 256], np.int64), 'restore'),
         nh.from_array(np.array([1/np.sqrt(32)], np.float32), 'scale'),
         nh.from_array(np.array(mask, np.float32), 'mask'),
         nh.from_array(np.eye(256, dtype=np.float32), 'projection')])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
    model.ir_version = 10
    return model


class LayoutAttention(unittest.TestCase):
    def test_fp32_fusion_handles_dynamic_batch_and_distinct_query_key_lengths(self):
        original = graph_model(0)
        candidate = graph_model(0)
        self.assertEqual(len(fuse_standard_attention(candidate)), 1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 4
        left = ort.InferenceSession(original.SerializeToString(), options, providers=['CPUExecutionProvider'])
        right = ort.InferenceSession(candidate.SerializeToString(), options, providers=['CPUExecutionProvider'])
        generator = np.random.default_rng(51)
        for batch, qlength, klength in [(1, 1, 1), (2, 3, 7)]:
            inputs = {name: generator.normal(size=(batch, qlength if name == 'q' else klength, 256)).astype(np.float32)
                      for name in ['q', 'k', 'v']}
            np.testing.assert_allclose(left.run(None, inputs)[0], right.run(None, inputs)[0], atol=2e-6, rtol=2e-5)

    def test_nonzero_attention_mask_is_not_dropped(self):
        with self.assertRaisesRegex(AssertionError, 'No verified'):
            fuse_standard_attention(graph_model(-10000))


if __name__ == '__main__':
    unittest.main()
