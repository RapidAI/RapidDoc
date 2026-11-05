"""Migration tools require an explicit source-tree opt-in."""
from pathlib import Path
import subprocess
import sys


def test_migration_tools_remain_executable_without_shipping_them():
    path=Path('scripts/migration/preview_compat.py').resolve()
    code='''import importlib.util,runpy,sys
import rapid_doc_rust as sdk
assert not hasattr(sdk,'parse_pdf_models')
assert importlib.util.find_spec('rapid_doc_rust._preview') is None
runpy.run_path(sys.argv[1])['activate']()
import rapid_doc_rust_preview as dev
assert sdk.OnnxSession is __import__('rapid_doc_rust._diagnostics',fromlist=['OnnxSession']).OnnxSession
assert dev.organize_table_detection_boxes([])==[]
assert dev.apply_table_ocr_results([],[])==[]
assert callable(sdk.parse_pdf_models)
assert isinstance(dev.compatibility_report()['modules'],list)
for name in dev.__all__:assert hasattr(dev,name),name
for name in ('prepare_native_document_containers','build_native_document_geometry_plan','apply_native_geometry_plan','prepare_native_document_geometry','prepare_native_page_marginals','prepare_native_document_sources'):
 assert not hasattr(dev,name),name
assert not any(name.startswith('rapid_doc._vendor') for name in sys.modules)
'''
    subprocess.run([sys.executable,'-I','-c',code,str(path)],capture_output=True,check=True)
