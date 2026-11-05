"""CUDA graph optimization is selected from originals and performed by Rust."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch,Mock


def runtime_module():
    path=Path(__file__).resolve().parents[2]/'bindings/python/python/rapid_doc_rust/_runtime.py'
    spec=importlib.util.spec_from_file_location('rapid_doc_rust._cuda_config_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class CudaConfig(unittest.TestCase):
    def test_all_optimized_roles_use_original_files_and_native_role(self):
        r=runtime_module();mapping={'det':'ocr_detection','rec':'ocr_recognition','seal_det':'ocr_detection',
            'layout':'layout','table':'slanet','wired':'unet','formula':'formula'}
        with patch.object(r,'prepare_cuda_runtime'),patch.object(r,'runtime_path',return_value=Path('runtime')), \
             patch.object(r,'resolve_model',side_effect=lambda role,override:(Path(role+'.onnx'),r.MANIFEST[role]['sha256'])), \
             patch.object(r,'operator_library',return_value=Path('operators')), \
             patch.dict(os.environ,{'RAPIDDOC_CUDA_MODEL_MANIFEST':'missing-obsolete-manifest.json'}):
            for role,optimization in mapping.items():
                model=r.LazySession(role);model.device='cuda';d=json.loads(model.descriptor())
                self.assertEqual((d['model'],d['optimization']),(role+'.onnx',optimization))
                self.assertNotIn('table_decoder',d)
                self.assertEqual(d['operator_libraries'],['operators']if role in ['det','rec','seal_det','wired','formula']else[])

    def test_cpu_custom_source_and_disable_do_not_rewrite(self):
        r=runtime_module()
        with patch.object(r,'prepare_cuda_runtime'),patch.object(r,'runtime_path',return_value=Path('runtime')), \
             patch.object(r,'operator_library',side_effect=AssertionError('No custom ops required')):
            for role in ['det','rec','seal_det','layout','table','wired','formula']:
                for device,sha,enabled in [('cpu',r.MANIFEST[role]['sha256'],True),('cuda','custom',True),('cuda',r.MANIFEST[role]['sha256'],False)]:
                    with patch.object(r,'resolve_model',return_value=(Path('original.onnx'),sha)):
                        model=r.LazySession(role);model.device=device;model.optimize=enabled
                        self.assertIsNone(json.loads(model.descriptor())['optimization'])
            with patch.object(r,'resolve_model',return_value=(Path('original.onnx'),r.MANIFEST['table']['sha256'])):
                model=r.LazySession('table');model.device='cuda';model.partition_table=False
                self.assertIsNone(json.loads(model.descriptor())['optimization'])

    def test_disable_flags_preserve_requested_cuda_device(self):
        r=runtime_module();cfg={'engine_cfg':{'use_cuda':True,'mask_uint8':False}}
        models=r.load_models(cfg,{'onnx_ops':False,'EngineConfig.onnxruntime.use_cuda':True},
            {'engine_cfg':{'use_cuda':True,'onnx_ops':False}},{'engine_cfg':{'use_cuda':True,'unet_conv_cache':False,'slanet_partition':False}})
        for role in ['detector','recognizer','seal_detector','seal_recognizer','layout_model','formula_model','wired_model']:
            self.assertEqual(models[role].device,'cuda');self.assertFalse(models[role].optimize)
        self.assertFalse(models['table_model'].partition_table)

    def test_reference_crop_profile_does_not_import_opencv(self):
        r=runtime_module()
        with patch.dict(sys.modules),patch.object(r.importlib.metadata,'version',return_value='4.13.0'):
            sys.modules.pop('cv2',None);self.assertEqual(r.reference_crop_sampling(),'opencv4');self.assertNotIn('cv2',sys.modules)
        r.reference_crop_sampling.cache_clear()
        with patch.object(r.importlib.metadata,'version',side_effect=r.importlib.metadata.PackageNotFoundError):self.assertEqual(r.reference_crop_sampling(),'opencv5')

    def test_cuda_libraries_preload_once_and_failed_setup_can_retry(self):
        r=runtime_module();fake=Mock();fake.preload_dlls.side_effect=[OSError('missing dependency'),None]
        with patch.dict(sys.modules,{'onnxruntime':fake}):
            with self.assertRaises(OSError):r.prepare_cuda_runtime()
            self.assertFalse(r._CUDA_RUNTIME_READY)
            r.prepare_cuda_runtime();r.prepare_cuda_runtime()
        self.assertEqual(fake.preload_dlls.call_count,2)

    def test_device_routing_and_invalid_requests(self):
        r=runtime_module();cfg={'engine_cfg':{'use_cuda':True}}
        models=r.load_models(cfg,cfg,cfg,cfg)
        self.assertTrue(all(v.device=='cuda'for v in models.values()if isinstance(v,r.LazySession)))
        self.assertTrue(all(v.device=='cpu'for v in r.load_models({},{},{},{}).values()if isinstance(v,r.LazySession)))
        for cfg,error in [({'use_cuda':'true'},ValueError),({'use_cuda':True,'gpu_id':1},NotImplementedError),({'use_npu':True},NotImplementedError)]:
            with self.assertRaises(error):r.load_models({'engine_cfg':cfg},{},{},{})

    def test_external_cuda_runtime_does_not_require_python_ort(self):
        import builtins
        r=runtime_module();original=builtins.__import__
        def without_ort(name,*args,**kwargs):
            if name=='onnxruntime':raise ModuleNotFoundError("No module named 'onnxruntime'",name='onnxruntime')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=without_ort):r.prepare_cuda_runtime()
        self.assertFalse(r._CUDA_RUNTIME_READY)

    def test_broken_installed_ort_is_not_silently_ignored(self):
        import builtins
        r=runtime_module();original=builtins.__import__
        def broken_ort(name,*args,**kwargs):
            if name=='onnxruntime':raise ModuleNotFoundError('Missing ORT dependency',name='numpy')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=broken_ort),self.assertRaises(ModuleNotFoundError):r.prepare_cuda_runtime()

    def test_raw_session_contract_and_device_cache_remain_distinct(self):
        import rapid_doc_rust
        r=runtime_module();cpu=r.LazySession('det');cuda=r.LazySession('det');cuda.device='cuda';created=[]
        def create(*args,**kw):created.append(kw);return object()
        with patch.object(r,'prepare_cuda_runtime'),patch.object(r,'resolve_model',return_value=(Path('model'),'sha')), \
             patch.object(r,'runtime_path',return_value=Path('runtime')),patch.object(rapid_doc_rust,'OnnxSession',side_effect=create):
            self.assertIsNot(cpu.session(),cuda.session());self.assertIs(cuda.session(),cuda.session())
        self.assertEqual([v['device']for v in created],['cpu','cuda'])
        self.assertTrue(all(not v.get('operator_libraries')for v in created))


if __name__=='__main__':unittest.main()
