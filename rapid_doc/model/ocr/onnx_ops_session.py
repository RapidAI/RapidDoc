"""Optional CUDA custom-op sessions; ordinary CPU OCR keeps its original path."""

import warnings
from packaging.version import Version


def configure_cuda_onnx(params, device, rapidocr_version):
    """Select CUDA/device for explicit ONNX engines, preserving user overrides."""
    if not device.startswith("cuda"):
        return
    names = [getattr(params.get(f"{task}.engine_type"), "value",
                     params.get(f"{task}.engine_type")) for task in ("Det", "Rec")]
    if "onnxruntime" not in names:
        return
    params.setdefault("EngineConfig.onnxruntime.use_cuda", True)
    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    field = "gpu_id" if Version(str(rapidocr_version)) < Version("3.4.3") else "cuda_ep_cfg.device_id"
    params.setdefault(f"EngineConfig.onnxruntime.{field}", gpu_id)


def ops_settings(value):
    if value is False or value is None:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("onnx_ops must be a bool or dict")
    defaults = {"enabled": True, "library": None, "cache_capacity": 64,
                "compact_ctc": True, "fallback": False, "preoptimize": True,
                "use_tf32": None, "conv_api": "classic"}
    if set(value) - defaults.keys():
        raise ValueError(f"Unknown onnx_ops options: {set(value) - defaults.keys()}")
    settings = {**defaults, **value}
    for key in ("enabled", "compact_ctc", "fallback", "preoptimize"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"onnx_ops.{key} must be a bool")
    capacity = settings["cache_capacity"]
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 0 <= capacity <= 128:
        raise ValueError("onnx_ops.cache_capacity must be an integer between 0 and 128")
    if settings["use_tf32"] is not None and not isinstance(settings["use_tf32"], bool):
        raise ValueError("onnx_ops.use_tf32 must be a bool or None")
    if settings["conv_api"] not in ("frontend", "classic"):
        raise ValueError("onnx_ops.conv_api must be frontend or classic")
    return settings if settings["enabled"] else None


class CompactCtcPrediction:
    """The two operations RapidOCR's existing CTC decoder reads from predictions."""

    def __init__(self, indices, scores):
        self.indices, self.scores = indices, scores

    def argmax(self, axis):
        if axis not in (2, -1):
            raise ValueError("Compact CTC predictions only support the character axis")
        return self.indices

    def max(self, axis):
        if axis not in (2, -1):
            raise ValueError("Compact CTC predictions only support the character axis")
        return self.scores


class CompactCtcSession:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, image):
        name = self.base.session.get_inputs()[0].name
        indices, scores = self.base.session.run(None, {name: image})
        return CompactCtcPrediction(indices, scores)


def install_onnx_ops(ocr_engine, settings):
    """Replace this OCR instance's CUDA sessions, preserving dictionary metadata.

    Construct every replacement before changing adapters. Caller-supplied custom
    sessions are left alone. Inference errors are never silently retried.
    """
    if settings is None:
        return {}
    eligible = []
    for task, attr in [("Det", "text_det"), ("Rec", "text_rec")]:
        cfg = getattr(ocr_engine.cfg, task)
        engine = getattr(cfg.engine_type, "value", cfg.engine_type)
        if engine != "onnxruntime" or cfg.get("session") is not None:
            continue
        adapter = getattr(ocr_engine, attr)
        base = adapter.session
        if base.session.get_providers()[0] != "CUDAExecutionProvider":
            continue
        eligible.append((task, cfg, adapter, base))
    if not eligible:
        return {}
    try:
        from rapid_doc.backend.onnx_ops import create_session, create_ctc_session
    except ImportError as error:
        message = "onnx_ops requires ONNX graph dependencies; reinstall RapidDoc with its dependencies"
        if not settings["fallback"]:
            raise RuntimeError(message) from error
        warnings.warn(message + "; using original OCR sessions", RuntimeWarning, stacklevel=2)
        return {}

    replacements = []
    for task, cfg, adapter, base in eligible:
        source = cfg.get("model_path")
        if source is None:
            raise ValueError(f"onnx_ops requires an explicit {task}.model_path")
        old = base.session
        provider_options = old.get_provider_options()
        providers = [(name, dict(provider_options.get(name, {}))) for name in old.get_providers()]
        if settings["use_tf32"] is not None:
            for name, options in providers:
                if name == "CUDAExecutionProvider":
                    options["use_tf32"] = int(settings["use_tf32"])
        compact = task == "Rec" and settings["compact_ctc"]
        factory = create_ctc_session if compact else create_session
        try:
            session = factory(source, providers=providers, sess_options=old.get_session_options(),
                              library=settings["library"], cache_capacity=settings["cache_capacity"],
                              use_tf32=settings["use_tf32"], conv_api=settings["conv_api"], preoptimize=settings["preoptimize"],
                              fallback=settings["fallback"])
        except Exception as error:
            if not settings["fallback"]:
                raise
            warnings.warn(f"onnx_ops {task} construction failed: {error}; using original OCR sessions",
                          RuntimeWarning, stacklevel=2)
            return {}
        replacements.append((task, adapter, base, session, compact))

    reports = {}
    for task, adapter, base, session, compact in replacements:
        # Replacing rather than retaining the old ORT object releases duplicate
        # weights/workspaces. Metadata is retained in the rewritten model.
        base.session = session
        if compact:
            adapter.session = CompactCtcSession(base)
        reports[task] = session.optimization_report
    return reports
