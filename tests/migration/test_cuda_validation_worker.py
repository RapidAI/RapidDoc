"""Explicit CPU helpers are distinct from a requested CUDA provider fallback."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2]/'scripts/migration/accept_cuda_pdf.py'
spec = importlib.util.spec_from_file_location('cuda_worker', path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_explicit_cpu_helper_is_allowed_in_gpu_pipeline():
    worker.verify_requested_provider(['CPUExecutionProvider'], ['CPUExecutionProvider'], True)


@pytest.mark.parametrize('requested', [None, [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']])
def test_cuda_fallback_is_still_rejected(requested):
    with pytest.raises(RuntimeError, match='fell back'):
        worker.verify_requested_provider(requested, ['CPUExecutionProvider'], True)
