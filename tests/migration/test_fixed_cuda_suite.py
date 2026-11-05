"""A differing Linux snapshot must produce diagnostics without Windows APIs."""
import importlib.util
from pathlib import Path
import sys


def test_snapshot_diagnostics_do_not_import_windows_sampler(monkeypatch):
    monkeypatch.setitem(sys.modules, 'accept_pdf_delivery', None)
    source = Path(__file__).resolve().parents[2] / 'scripts/migration/accept_fixed_cuda_suite.py'
    spec = importlib.util.spec_from_file_location('fixed_cuda_suite', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = list(module.differences({'pages': [{'text': 'old'}]}, {'pages': [{'text': 'new'}]}))
    assert rows == [dict(path='$.pages[0].text', kind='value', expected="'old'", actual="'new'")]
    assert list(module.differences({'same': []}, {'same': []})) == []
