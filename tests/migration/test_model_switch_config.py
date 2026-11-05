"""Model switches resolve concrete artifacts without altering default choices."""
import unittest
from test_cuda_runtime_config import runtime_module

class ModelSwitching(unittest.TestCase):
    def test_layout_formula_and_table_selection(self):
        r=runtime_module()
        for layout in ['pp_doclayout_plus_l','pp_doclayoutv2','pp_doclayoutv3']:
            for formula in ['pp_formulanet_plus_s','pp_formulanet_plus_m','pp_formulanet_plus_l']:
                models=r.load_models({'model_type':layout},{},{'model_type':formula},{'model_type':'table_transformer','rule_table_engine':'docvortex_ocr'})
                self.assertEqual(models['layout_model'].model_id,'layout'if layout=='pp_doclayoutv3'else layout)
                self.assertEqual(models['formula_model'].model_id,'formula'if formula=='pp_formulanet_plus_m'else formula)
                self.assertEqual(models['table_model'].model_id,'table_transformer_v1_1_all')
                self.assertEqual(models['table_variant'],'table-transformer')

    def test_original_defaults_and_custom_paths_are_preserved(self):
        r=runtime_module();models=r.load_models({},{},{},{})
        self.assertEqual(models['layout_model'].model_id,'layout')
        self.assertEqual(models['formula_model'].model_id,'formula')
        self.assertEqual(models['table_variant'],'unet-slanet-plus')
        models=r.load_models({'model_type':'pp_doclayoutv2','model_dir_or_path':'custom-layout.onnx'},
            {},{'model_type':'pp_formulanet_plus_l','model_dir_or_path':'custom-formula.onnx'},
            {'model_type':'table_transformer','model_dir_or_path':'custom-table.onnx'})
        self.assertEqual(models['layout_model'].override,'custom-layout.onnx')
        self.assertEqual(models['formula_model'].override,'custom-formula.onnx')
        self.assertEqual(models['table_model'].override,'custom-table.onnx')

    def test_optimizations_do_not_misidentify_variant_models(self):
        r=runtime_module()
        for role,variant in [('layout','pp_doclayoutv2'),('layout','pp_doclayout_plus_l'),('formula','pp_formulanet_plus_s'),('formula','pp_formulanet_plus_l'),('table','table_transformer_v1_1_all')]:
            self.assertIsNone(r.native_optimization(role,r.CATALOG['models'][variant]['sha256'],'cuda'))

    def test_fallback_is_lazy_and_independently_configurable(self):
        from unittest.mock import patch
        r=runtime_module()
        with patch.object(r,'resolve_model',side_effect=AssertionError('No download during configuration')):
            default=r.load_models({},{},{},{})
            self.assertIsNone(default['fallback_table_variant'])
            self.assertNotIn('fallback_table_model',default)
            for name,variant in [('table_transformer','table-transformer'),('unet_slanet_plus','unet-slanet-plus'),('unet','unet'),('slanet_plus','slanet-plus')]:
                models=r.load_models({},{},{},{'fallback_model_type':name,'fallback_to_text':True})
                self.assertEqual(models['fallback_table_variant'],variant)
                self.assertEqual(models['fallback_table_model'].model_id,'table_transformer_v1_1_all'if name=='table_transformer'else'wired'if name=='unet'else'table')
            r.load_models({},{},{},{'fallback_to_text':True})
        for cfg in [{'fallback_text_loss_threshold':float('nan')},{'fallback_text_loss_threshold':-1},{'fallback_text_loss_threshold':True},{'fallback_model_type':'missing'},{'fallback_to_text':1}]:
            with self.assertRaises(ValueError):r.load_models({},{},{},cfg)
