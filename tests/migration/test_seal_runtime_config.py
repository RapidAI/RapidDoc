"""Seal delivery contracts: lazy resolution and independent default weights."""
import hashlib
import os
from pathlib import Path
import sys
import unittest

if os.environ.get('RAPIDDOC_TEST_SDK'):
    sys.path.insert(0, os.environ['RAPIDDOC_TEST_SDK'])


class SealRuntimeConfig(unittest.TestCase):
    def models(self, ocr=None):
        from rapid_doc_rust._runtime import load_models
        return load_models({}, ocr or {}, {}, {})

    def test_seal_sessions_are_lazy(self):
        from rapid_doc_rust._runtime import LazySession
        models = self.models()
        self.assertEqual(models['seal_detector'].role, 'seal_det')
        self.assertEqual(models['seal_recognizer'].role, 'rec')
        self.assertTrue(all(model._session is None for model in models.values() if isinstance(model, LazySession)))

    def test_disabled_seals_do_not_load_models(self):
        models = self.models({'seal_enable': False})
        self.assertIsNone(models['seal_detector']._session)
        self.assertIsNone(models['seal_recognizer']._session)

    def test_body_recognizer_override_does_not_replace_seal_weights(self):
        path = 'custom-body-recognizer.onnx'
        models = self.models({'Rec.model_path': path})
        self.assertEqual(models['recognizer'].override, path)
        self.assertIsNone(models['seal_recognizer'].override)

    def test_seal_model_manifest_matches_original_resource(self):
        from rapid_doc_rust._runtime import MANIFEST, resolve_model
        path, sha = resolve_model('seal_det')
        self.assertEqual(path.name, 'pp-ocrv4_mobile_seal_det.onnx')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)
        self.assertEqual(sha, MANIFEST['seal_det']['sha256'])
        self.assertEqual(sha, 'e6109a1022b5ebf0822fc00646ef2398a7ef387390ca5c978de79352b1314204')


if __name__ == '__main__':
    unittest.main()
