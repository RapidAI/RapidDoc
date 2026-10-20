import ctypes
import json
import os
import sys
import threading
from pathlib import Path

_handles = {}
_lock = threading.Lock()
_dependencies = []
_dll_directories = []

def _preload():
    # Resolve NVIDIA pip libraries without importing PyTorch. System libraries
    # remain supported through the normal OS loader search path.
    roots = [Path(p) / 'nvidia' for p in sys.path if p]
    for component, pattern in [('cuda_runtime','cudart64*.dll' if os.name=='nt' else 'libcudart.so.12'),
                               ('cublas','cublasLt64_12.dll' if os.name=='nt' else 'libcublasLt.so.12'),
                               ('cublas','cublas64_12.dll' if os.name=='nt' else 'libcublas.so.12'),
                               ('cudnn','cudnn64_9.dll' if os.name=='nt' else 'libcudnn.so.9')]:
        for root in roots:
            for sub in ('lib','bin'):
                directory=root/component/sub
                files=sorted(directory.glob(pattern))
                if files:
                    if os.name=='nt':_dll_directories.append(os.add_dll_directory(str(directory)))
                    _dependencies.append(ctypes.CDLL(str(files[0]),mode=ctypes.RTLD_GLOBAL))
                    break
            else:continue
            break
def library_path(path=None):
    selected = path or os.environ.get("ONNXRUNTIME_OPS_LIBRARY")
    if selected:
        candidate = Path(selected).expanduser().resolve()
        if not candidate.is_file(): raise FileNotFoundError(candidate)
        return candidate
    name = "onnxruntime_ops.dll" if os.name == "nt" else "libonnxruntime_ops.so"
    candidate = Path(__file__).parent / "lib" / name
    if not candidate.is_file(): raise FileNotFoundError("Native plugin not found; build it or set ONNXRUNTIME_OPS_LIBRARY")
    return candidate

def _handle(path=None):
    p = str(library_path(path))
    with _lock:
        if p not in _handles:
            if not _handles:_preload()
            _handles[p] = ctypes.CDLL(p)
    return _handles[p]

def build_info(path=None):
    fn = _handle(path).OrtOpsBuildInfo; fn.restype = ctypes.c_char_p
    return json.loads(fn().decode())

def stats(path=None):
    fn = _handle(path).OrtOpsStats; fn.restype = ctypes.c_char_p
    return json.loads(fn().decode())
