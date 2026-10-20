"""Create a session without changing the caller's original model file."""
import tempfile
import warnings
import weakref
import threading
import os
from pathlib import Path
import onnx
import onnxruntime as ort
from .native import library_path, build_info
from .rewrite import RewriteReport, rewrite_model

_registered = weakref.WeakKeyDictionary()
_registration_lock = threading.Lock()

def _register(options, lib):
    with _registration_lock:
        libraries = _registered.setdefault(options, set())
        key = str(lib.resolve())
        if key not in libraries:
            options.register_custom_ops_library(key)
            libraries.add(key)

def _source(source):
    if isinstance(source, onnx.ModelProto): return source.SerializeToString()
    if isinstance(source, (str, Path)): return str(source)
    return source

def create_session(model, *, library=None, providers=None, sess_options=None,
                   cache_capacity=64, use_tf32=None, conv_api="classic", conv=True, reductions=True,
                   preoptimize=True, fallback=True):
    """Keep outputs unchanged; only use plugin nodes when CUDA is requested.

    fallback covers plugin loading/session construction only, never inference errors.
    preoptimize creates a temporary official-ORT optimized graph to retain existing
    constant folding/fusions; it does not persist another user model.
    """
    source = _source(model)
    if not isinstance(cache_capacity, int) or not 0 <= cache_capacity <= 128:
        raise ValueError("cache_capacity must be an integer between 0 and 128")
    if conv_api not in ("frontend", "classic"):
        raise ValueError("conv_api must be frontend or classic")
    requested = providers or ["CPUExecutionProvider"]
    cuda = next((p for p in requested if (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"), None)
    if use_tf32 is None:
        value = cuda[1].get("use_tf32", 1) if isinstance(cuda, tuple) else 1
        use_tf32 = str(value).lower() not in ("0", "false")
    if isinstance(cuda, tuple) and (str(cuda[1].get("prefer_nhwc", 0)) == "1" or str(cuda[1].get("fuse_conv_bias", 0)) == "1"):
        raise ValueError("Plugin currently requires NCHW and fuse_conv_bias=0")
    if isinstance(cuda, tuple) and str(cuda[1].get("enable_cuda_graph", 0)) == "1":
        raise ValueError("CUDA Graph capture is not supported by this plugin")
    def standard(reason):
        session = ort.InferenceSession(source, sess_options=sess_options, providers=requested)
        session.optimization_report = RewriteReport(reason=reason)
        return session
    if cuda is None: return standard("CPU or non-CUDA providers requested")
    try:
        version = tuple(int(v) for v in ort.__version__.split("+")[0].split(".")[:2])
        if version < (1, 16): raise RuntimeError("ORT >= 1.16 required")
        lib = library_path(library); info = build_info(lib)
        if conv and conv_api == "classic" and not info.get("classic_conv"):
            raise RuntimeError("Classic convolution requires the updated native library (0.2.0+)")
        if conv and version < (1, 20) and (use_tf32 or os.environ.get('NVIDIA_TF32_OVERRIDE') != '0'):
            warnings.warn("Legacy ORT convolution and cuDNN frontend can differ with TF32. "
                          "For validated FP32 OCR equivalence, set NVIDIA_TF32_OVERRIDE=0 before startup "
                          "and CUDA provider use_tf32=0; validate your model outputs.", RuntimeWarning, stacklevel=2)
        if preoptimize:
            options = ort.SessionOptions()
            if sess_options:
                for name in ("intra_op_num_threads", "inter_op_num_threads", "graph_optimization_level", "execution_mode", "enable_mem_pattern", "enable_cpu_mem_arena"):
                    setattr(options, name, getattr(sess_options, name))
            options.log_severity_level = 3
            with tempfile.TemporaryDirectory(prefix="onnxruntime-ops-") as directory:
                options.optimized_model_filepath = str(Path(directory) / "optimized.onnx")
                base = ort.InferenceSession(source, sess_options=options, providers=requested)
                if "CUDAExecutionProvider" not in base.get_providers(): raise RuntimeError("CUDA provider failed to load")
                proto = onnx.load(options.optimized_model_filepath); del base
        else:
            proto = onnx.load(source) if isinstance(source, str) else onnx.load_model_from_string(source)
            onnx.external_data_helper.convert_model_from_external_data(proto)
        rewritten, report = rewrite_model(proto, cache_capacity=cache_capacity, use_tf32=use_tf32, conv_api=conv_api, conv=conv, reductions=reductions, validate=not preoptimize)
        if not report.enabled: return standard("No supported patterns")
        options = sess_options or ort.SessionOptions()
        _register(options, lib)
        session = ort.InferenceSession(rewritten.SerializeToString(), sess_options=options, providers=requested)
        if "CUDAExecutionProvider" not in session.get_providers(): raise RuntimeError("CUDA provider failed to load")
        session.optimization_report = report
        return session
    except Exception as error:
        if not fallback: raise
        warnings.warn(f"onnxruntime-ops disabled: {error}; using the original graph", RuntimeWarning, stacklevel=2)
        return standard(str(error))
