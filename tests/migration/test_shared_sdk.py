"""Shared SDK ownership, lazy models and Python compatibility boundaries."""
import subprocess
import sys


def run(code):
    subprocess.run([sys.executable,'-c',code],capture_output=True,check=True)


def test_warmup_and_parsing_share_native_sessions():
    run('''import json,sys
from pathlib import Path
from rapid_doc_rust import RapidDoc
from rapid_doc_rust._runtime import prepare_device
engine=RapidDoc(layout_config={'model_type':'docvortex_rule'},ocr_config={'seal_enable':False},formula_enable=False,table_enable=False,preload_model=True)
native=engine._native_engine
before=native.loaded_model_roles()
assert before==['detector','recognizer'],before
events=[]
native.warmup('{}',events.append,prepare_device)
assert not events,events
result=engine(Path('tests/migration/inputs/pdf_native.pdf'))
assert result.markdown
assert native is engine._native_engine
assert native.loaded_model_roles()==before
engine.layout_config['conf_thresh']=.25
assert engine._get_native_engine() is native
assert native.loaded_model_roles()==before
engine.ocr_config['engine_cfg']={'cpu_spinning':False}
assert engine._get_native_engine() is native
assert native.loaded_model_roles()==[]
assert 'rapid_doc_rust._diagnostics' not in sys.modules
assert 'rapid_doc_rust_preview' not in sys.modules
''')


def test_public_api_does_not_reenter_python_model_or_text_pipeline():
    run('''from pathlib import Path
from unittest.mock import patch
from rapid_doc_rust import RapidDoc
engine=RapidDoc(layout_config={'model_type':'docvortex_rule'},formula_enable=False,table_enable=False)
with patch.object(engine,'_get_models',side_effect=AssertionError('Python model orchestration used')),patch('rapid_doc_rust.parse_pdf_text',side_effect=AssertionError('Duplicate Python page counting')):
 result=engine(Path('tests/migration/inputs/pdf_native.pdf'),end_page_id=999)
assert len(result.middle_json['pdf_info'])==1
try:engine(Path('tests/migration/inputs/pdf_native.pdf'),start_page_id=1)
except ValueError:pass
else:raise AssertionError('Invalid start was accepted')
''')


def test_office_is_lazy_even_with_invalid_unused_model_settings():
    run('''import json
from pathlib import Path
from rapid_doc_rust import RapidDoc
engine=RapidDoc(layout_config={'model_type':'unknown-unused-model'})
result=engine(Path('tests/migration/inputs/synthetic.docx'))
assert result.markdown
assert engine._native_engine.loaded_model_roles()==[]
''')


def test_call_overrides_use_shared_configuration():
    run('''import json
from rapid_doc_rust._native import model_plan_json
from rapid_doc_rust._runtime import model_config
plan=json.loads(model_plan_json(model_config({'model_type':'pp_doclayoutv2'},{'engine_cfg':{'use_cuda':False}},{'model_type':'pp_formulanet_plus_s'},{'model_type':'table_transformer','fallback_model_type':'unet_slanet_plus'})))
assert plan['models']['layout_model']['model_id']=='pp_doclayoutv2'
assert plan['models']['formula_model']['model_id']=='pp_formulanet_plus_s'
assert plan['models']['table_model']['model_id']=='table_transformer_v1_1_all'
assert plan['fallback_table_variant']=='unet-slanet-plus'
''')


def test_explicit_null_cpu_overrides_and_invalid_model_types():
    run('''import json
from rapid_doc_rust._runtime import load_models
models=load_models({'engine_cfg':{'cpu_spinning':None,'cpu_dynamic_block_base':None}},{},{},{},cpu_spinning=True,cpu_dynamic_block_base=8)
assert models['layout_model'].cpu_spinning is None
assert models['layout_model'].cpu_dynamic_block_base is None
assert models['detector'].cpu_spinning is True
for config in ({'model_type':None},{'engine_type':False}):
 try:load_models(config,{},{},{})
 except NotImplementedError:pass
 else:raise AssertionError(config)
''')
