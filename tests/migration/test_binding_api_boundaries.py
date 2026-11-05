"""The default SDK avoids preview algorithms and preserves explicit diagnostic access."""
import subprocess
import sys
import pytest
from rapid_doc_rust import _native


def run(code):
    subprocess.run([sys.executable, '-c', code], capture_output=True, check=True)


def test_production_office_does_not_import_preview_or_drawing():
    run('''import sys
from pathlib import Path
from rapid_doc_rust import RapidDoc
engine=RapidDoc()
assert 'rapid_doc_rust._preview' not in sys.modules
result=engine(Path('tests/migration/inputs/synthetic.docx'))
assert result.markdown and result.middle_json and result.content_list_json
assert 'rapid_doc_rust._preview' not in sys.modules
assert 'rapid_doc_rust._drawing' not in sys.modules
''')


def test_supported_diagnostics_are_lazy_and_migration_apis_are_not_shipped():
    run('''import sys,importlib.util
import rapid_doc_rust as sdk
assert 'rapid_doc_rust._diagnostics' not in sys.modules
assert 'OnnxSession' in dir(sdk)
assert 'rapid_doc_rust._diagnostics' not in sys.modules
for name in sdk.__all__:assert getattr(sdk,name) is not None,name
assert 'rapid_doc_rust._diagnostics' in sys.modules
assert 'pdf' in sdk.capabilities()['formats']
assert importlib.util.find_spec('rapid_doc_rust._preview') is None
assert importlib.util.find_spec('rapid_doc_rust._native_pdf') is None
assert not hasattr(sdk,'parse_pdf_models')
assert not hasattr(sdk,'_recognize_page_png')
assert 'rapid_doc_rust_preview' not in sys.modules
''')


def test_production_pdf_keeps_preview_and_drawings_unloaded():
    run('''import sys
from pathlib import Path
from rapid_doc_rust import RapidDoc
engine=RapidDoc(layout_config={'model_type':'docvortex_rule'},formula_enable=False,table_enable=False)
result=engine(Path('tests/migration/inputs/pdf_native.pdf'))
assert result.markdown and result.content_list_json
assert 'rapid_doc_rust._preview' not in sys.modules
assert 'rapid_doc_rust._drawing' not in sys.modules
''')


def test_production_progress_preserves_original_callback_exception():
    run('''import json
from pathlib import Path
from rapid_doc_rust._native import DocumentEngine
from rapid_doc_rust._runtime import resource_hints
data=Path('tests/migration/inputs/pdf_native.pdf').read_bytes()
class Interrupted(Exception):pass
seen=[]
def progress(event):
    assert isinstance(event,dict)
    assert 'stage' in event and 'page_index' in event and 'detail' in event
    seen.append(event)
    raise Interrupted('stop callback')
try:
    DocumentEngine(json.dumps({'formula_enable':False,'table_enable':False}),resource_hints(pdfium=True)).parse(data,'{}',progress,lambda device:None)
except Interrupted as e:assert str(e)=='stop callback'
else:raise AssertionError('callback exception was lost')
assert seen
''')


@pytest.mark.skipif(getattr(_native,'MIGRATION_TOOLS',False),reason='Checks the default release artifact')
def test_release_native_excludes_private_migration_checkpoints():
    run('''from rapid_doc_rust import _native
assert not _native.MIGRATION_TOOLS
for name in ('PdfEngine','parse_docx_bytes','parse_office_bytes','read_pdf_character_primitives_json','prepare_model_crops_bytes','FormulaDecoder','_PageImage'):
    assert not hasattr(_native,name),name
''')
