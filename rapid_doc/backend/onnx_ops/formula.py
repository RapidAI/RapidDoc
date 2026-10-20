"""Optional CUDA acceleration of standard formula models, entirely in memory."""
import warnings

import onnx
import onnxruntime as ort

from .formula_rewrite import optimize
from .native import library_path
from .session import _register, _source


def create_formula_session(model, *, providers=None, sess_options=None,
                           enabled=True, library=None, fallback=True):
    """Rewrite decoder ArgMax only; retain original outputs and model metadata.

    CPU/non-CUDA and disabled paths do not parse/rewrite the graph or load the native library.
    Fallback covers construction failures, never inference failures.
    """
    source = _source(model)
    requested = providers or ["CPUExecutionProvider"]

    def standard(reason):
        session = ort.InferenceSession(source, sess_options=sess_options, providers=requested)
        session.optimization_report = {"enabled": False, "reason": reason}
        return session

    cuda = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
               for p in requested)
    if not enabled or not cuda:
        return standard("Disabled or non-CUDA providers requested")
    try:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        lib = library_path(library)
        proto = (onnx.load(source) if isinstance(source, str)
                 else onnx.load_model_from_string(source))
        onnx.external_data_helper.convert_model_from_external_data(proto)
        report = optimize(proto, "maxindex")
        if not report["argmax_rewrites"]:
            return standard("No supported FP32 decoder ArgMax; " + str(report["skipped_argmax"]))
        options = sess_options or ort.SessionOptions()
        _register(options, lib)
        session = ort.InferenceSession(proto.SerializeToString(), sess_options=options,
                                       providers=requested)
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA provider failed to load")
        session.optimization_report = dict(report, enabled=True)
        return session
    except Exception as error:
        if not fallback:
            raise
        warnings.warn(f"Formula CUDA ops disabled: {error}; using the original graph",
                      RuntimeWarning, stacklevel=2)
        return standard(str(error))
