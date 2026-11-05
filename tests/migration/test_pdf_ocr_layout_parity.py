"""Layout suppression must use the same polygon evidence as Python."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rapid_doc.main import RapidDoc  # Initialize the original pipeline import graph.
from rapid_doc.backend.utils.utils import filter_overlap_boxes
from rapid_doc_rust import legacy_pipeline_json
from rapid_doc_rust._host_context import host_call
from rapid_doc_rust._runtime import geos_runtime


class LayoutOverlapParity(unittest.TestCase):
    def test_body_default_configuration_is_accepted_without_loading_sessions(self):
        from rapid_doc_rust._runtime import load_models, LazySession
        models = load_models({}, {"Det.box_thresh": 0.3}, {}, {})
        self.assertTrue(all(m._session is None for m in models.values() if isinstance(m, LazySession)))

    def test_page_batch_reaches_shared_engine_without_silent_truncation(self):
        from rapid_doc_rust import RapidDoc as RustDoc
        class Captured(Exception):
            pass
        class Probe:
            def __init__(self,config,resources):self.config=json.loads(config)
            def set_resources(self,*args):pass
            def set_config(self,*args):pass
            def parse(self,data,options,*callbacks):
                self.options=json.loads(options)
                raise Captured()
        engine = RustDoc(parse_method="ocr", pdf_pages_batch=64)
        with patch('rapid_doc_rust._native.DocumentEngine',Probe),self.assertRaises(Captured):
            engine(b"%PDF-1.7\n")
        self.assertEqual(engine._native_engine.config['pdf_pages_batch'],64)
        self.assertEqual(engine._native_engine.options['parse_method'],'ocr')
        self.assertIsNone(engine._native_engine.options['end_page'])

    def test_concave_polygon_preserves_region_inside_rectangle_gap(self):
        a = dict(original_label="vision_footnote", original_order=0,
                 poly=[4, 1, 10, 1, 10, 7, 4, 7],
                 polygon_points=[[4, 1], [10, 1], [10, 7], [4, 7]])
        b = dict(original_label="table", original_order=1,
                 poly=[0, 0, 10, 0, 10, 10, 0, 10],
                 polygon_points=[[0, 0], [2, 0], [2, 8], [10, 8], [10, 10], [0, 10]])
        @host_call
        def native(boxes):
            return json.loads(legacy_pipeline_json("filter_layout", json.dumps(dict(
                layout_dets=boxes, use_custom_ocr=False, geos_library=geos_runtime()))))
        for boxes in ([a, b], [b, a], [a, dict(a, original_order=2)]):
            with self.subTest(boxes=boxes):
                self.assertEqual(native(boxes), filter_overlap_boxes(boxes, False))
        self.assertEqual(len(native([a, b])), 2)
        self.assertEqual(len(native([a, dict(a, original_order=2)])), 1)


if __name__ == "__main__":
    unittest.main()
