"""Run against an installed wheel and a prepared Windows private ORT bundle."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows activation contexts")
def test_context_restored_after_failed_and_threaded_session_creation(tmp_path):
    runtime = os.environ.get("RAPIDDOC_TEST_PRIVATE_ORT")
    if not runtime:
        pytest.skip("Set RAPIDDOC_TEST_PRIVATE_ORT to the packaged private ORT DLL")
    model = Path("rapid_doc/resources/PP-OCRv6_rec_small.onnx").resolve()
    if not model.is_file():
        pytest.skip("Local OCR recognition model not installed")
    # A child process owns its ORT environment, independent of other test runtimes.
    code = r"""
import concurrent.futures, ctypes, hashlib, sys
from ctypes import wintypes
from pathlib import Path
from rapid_doc_rust import OnnxSession

runtime, model, invalid = map(Path, sys.argv[1:])
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentActCtx.argtypes = [ctypes.POINTER(wintypes.HANDLE)]
kernel.GetCurrentActCtx.restype = wintypes.BOOL
kernel.ReleaseActCtx.argtypes = [wintypes.HANDLE]

def current():
    handle = wintypes.HANDLE()
    if not kernel.GetCurrentActCtx(ctypes.byref(handle)):
        raise ctypes.WinError(ctypes.get_last_error())
    value = handle.value
    if value:
        kernel.ReleaseActCtx(handle)
    return value

before = current()
invalid.write_bytes(b"not a protobuf model")
try:
    OnnxSession(runtime, invalid, hashlib.sha256(invalid.read_bytes()).hexdigest(), threads=1)
except Exception as error:
    assert "Protobuf" in str(error), str(error)
else:
    raise AssertionError("Invalid model accepted")
assert current() == before, "Context leaked on error"

digest = hashlib.sha256(model.read_bytes()).hexdigest()
def load(_):
    before = current()
    session = OnnxSession(runtime, model, digest, threads=1)
    assert current() == before, "Context leaked on success"

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(load, range(4)))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(runtime).resolve()),
            str(model),
            str(tmp_path / "bad.onnx"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
