"""Installed production API has no invented document capacity defaults."""
import inspect
import io
import json
import os
import unittest
from unittest.mock import patch

import rapid_doc_rust as sdk
from rapid_doc_rust._host_context import host_call


class ProductionResourcePolicy(unittest.TestCase):
    def test_input_is_unbounded_unless_requested(self):
        class Stream(io.BytesIO):
            def read(self, *args):
                self.requested = args
                return super().read(*args)
        stream = Stream(b"complete input")
        self.assertEqual(sdk._read_bytes(stream, None), b"complete input")
        self.assertEqual(stream.requested, ())
        self.assertFalse(stream.closed)
        stream = Stream(b"complete input")
        self.assertEqual(sdk._read_bytes(stream, 2), b"com")
        self.assertEqual(stream.requested, (3,))
        for name in ("max_input_bytes", "max_pages", "max_chars_per_page", "max_total_chars"):
            self.assertIsNone(inspect.signature(sdk.parse_pdf_text).parameters[name].default)

    def test_production_formula_text_keeps_all_characters_and_restores_scope(self):
        from rapid_doc_rust._native import fix_formula_text
        text = "x+y " * 20_000
        @host_call
        def run():
            return fix_formula_text(text)
        self.assertEqual(run(), text)
        with self.assertRaises(sdk.RapidDocError):
            fix_formula_text(text)  # Diagnostic entry remains independently bounded.

    def test_explicit_resources_are_enforced_and_restored(self):
        @host_call
        def run():
            return json.loads(sdk.legacy_pipeline_json("filter_layout", '{"layout_dets":[],"use_custom_ocr":false}'))
        for limit in ('{"max_json_nodes":1}', '{"max_json_bytes":2}'):
            with patch.dict(os.environ, {"RAPIDDOC_RESOURCE_LIMITS": limit}):
                with self.assertRaises(sdk.RapidDocError):
                    run()
        self.assertEqual(run(), [])

    def test_formula_batch_has_no_added_upper_bound(self):
        self.assertIsNone(sdk.RapidDoc(formula_config={"batch_num": 65})._models)
        with self.assertRaises(ValueError):
            sdk.RapidDoc(formula_config={"batch_num": 0})

    def test_page_count_above_old_ten_thousand_default_and_explicit_limit(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        for _ in range(10_001):
            writer.add_blank_page(72, 72)
        stream = io.BytesIO()
        writer.write(stream)
        data = stream.getvalue()
        result = sdk.parse_pdf_text(data, start_page=0, end_page=0, allow_partial=True)
        self.assertEqual(result.document["source_page_count"], 10_001)
        with self.assertRaises(sdk.RapidDocError):
            sdk.parse_pdf_text(data, start_page=0, end_page=0, allow_partial=True, max_pages=10_000)


if __name__ == "__main__":
    unittest.main()
