"""Bindings delegate original-model resolution to the shared Rust store."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from test_cuda_runtime_config import runtime_module

class OriginalModels(unittest.TestCase):
    def test_directory_environment_priority(self):
        r=runtime_module()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'RAPID_MODELS_DIR':d,'RAPIDDOC_MODEL_DIR':'unused'}):
            self.assertEqual(r.download_directory('formula'),Path(d).resolve())
            self.assertEqual(r.download_directory('det'),Path(d).resolve())
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'RAPID_MODELS_DIR':'','RAPIDDOC_MODEL_DIR':d}):
            self.assertEqual(r.download_directory(),Path(d).resolve())

    def test_standalone_directory_does_not_inspect_python_package(self):
        r=runtime_module()
        with patch.dict(os.environ,{'RAPID_MODELS_DIR':'','RAPIDDOC_MODEL_DIR':''}), \
             patch.object(r.importlib.util,'find_spec',side_effect=AssertionError('No package scan')):
            self.assertEqual(r.download_directory(),Path.home()/'.cache/rapid_doc/models')

    def test_catalog_pins_defaults_and_separates_variants(self):
        r=runtime_module()
        self.assertEqual(r.CATALOG['defaults']['layout'],'pp_doclayoutv3')
        self.assertEqual(r.CATALOG['defaults']['formula'],'pp_formulanet_plus_m')
        self.assertEqual(r.CATALOG['defaults']['table'],'slanet_plus')
        for model in r.CATALOG['models'].values():
            self.assertEqual(set(model),{'filename','url','sha256'})
            self.assertTrue(model['url'].startswith('https://www.modelscope.cn/'))
        self.assertIn('table_transformer_v1_1_all',r.CATALOG['models'])
        self.assertIn('pp_doclayoutv2',r.CATALOG['models'])
        self.assertIn('pp_formulanet_plus_s',r.CATALOG['models'])
        self.assertIn('pp_formulanet_plus_l',r.CATALOG['models'])

    def test_custom_override_uses_own_hash_and_no_python_discovery(self):
        r=runtime_module()
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'custom.onnx';target.write_bytes(b'custom')
            with patch.object(r.importlib.util,'find_spec',side_effect=AssertionError('No package scan')):
                self.assertEqual(r.resolve_model('layout',target),(target,hashlib.sha256(b'custom').hexdigest()))
            with self.assertRaises(FileNotFoundError):r.resolve_model('layout',Path(d)/'missing.onnx')

    def test_binding_only_delegates_resolution(self):
        r=runtime_module()
        with patch.dict(os.environ,{'RAPID_MODELS_DIR':'shared-models'}), \
             patch.object(r,'resolve_model_json',return_value=json.dumps(dict(path='model.onnx',sha256='hash',model_id='pp_doclayoutv3',downloaded=False))) as native:
            self.assertEqual(r.resolve_model('layout'),(Path('model.onnx'),'hash'))
            native.assert_called_once_with('layout',directory='shared-models',override_path=None)

    def test_packaged_operator_library_needs_no_original_python_package(self):
        r=runtime_module()
        with patch.dict(os.environ,{'RAPIDDOC_OPERATOR_LIBRARY':'','ONNXRUNTIME_OPS_LIBRARY':''}), \
             patch.object(r.importlib.util,'find_spec',side_effect=AssertionError('No original Python import')):
            path=r.operator_library()
            self.assertTrue(path.is_file());self.assertIn('rapid_doc_rust',str(path))

if __name__=='__main__':unittest.main()
