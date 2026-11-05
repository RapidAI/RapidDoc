"""Automatic local runtime/model discovery and reusable, lazy CPU sessions."""

import importlib.util
import importlib.metadata
from functools import lru_cache
import json
import os
import sys
import threading
from pathlib import Path
from loguru import logger

from ._native import model_catalog_json, model_directory_path, resolve_model_json

CATALOG = json.loads(model_catalog_json())
# Existing inference roles retain their pinned original weights and defaults.
MANIFEST = {role: CATALOG["models"][model_id] for role, model_id in CATALOG["defaults"].items()}
_SESSIONS = {}
_LOCK = threading.RLock()
_CUDA_RUNTIME_READY = False


@lru_cache(maxsize=1)
def reference_crop_sampling():
    """Match the environment's Python reference without loading OpenCV itself."""
    loaded = sys.modules.get("cv2")
    if loaded is not None:
        return "opencv4" if str(loaded.__version__).split(".")[0] == "4" else "opencv5"
    for name in (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        return "opencv4" if version.split(".")[0] == "4" else "opencv5"
    return "opencv5"


def prepare_cuda_runtime():
    """Load the environment's NVIDIA libraries before native CUDA sessions.

    The original Python parser imports libraries from NVIDIA site packages;
    relying only on the OS loader can select a different cuBLAS/cuDNN version.
    This is once-per-process runtime setup, with no model/page work or PyTorch
    dependency. Explicit native descriptors remain available to other bindings.
    """
    global _CUDA_RUNTIME_READY
    with _LOCK:
        if not _CUDA_RUNTIME_READY:
            try:
                import onnxruntime
            except ModuleNotFoundError as exc:
                if exc.name != "onnxruntime":
                    raise
                # An external native ORT/CUDA installation needs no Python
                # wrapper. Let Rust discover it and validate its providers.
                return
            onnxruntime.preload_dlls(directory="")
            _CUDA_RUNTIME_READY = True


def operator_library():
    from ._native import locate_runtime

    return Path(locate_runtime(resource_hints(include_ort=False), "operators"))


def native_optimization(role, sha256, device, enabled=True):
    from ._native import model_optimization_json

    return json.loads(model_optimization_json(role, sha256, device, enabled))


def runtime_path():
    from ._native import locate_runtime

    return Path(locate_runtime(resource_hints(), "ort"))


def _configured_model_directory():
    # Pass Python's current environment explicitly: os.environ updates on
    # Windows need not be reflected in another CRT's native environment.
    return os.environ.get("RAPID_MODELS_DIR") or os.environ.get("RAPIDDOC_MODEL_DIR") or None


def download_directory(role=None):
    return Path(model_directory_path(_configured_model_directory()))


def resolve_model(role, override=None):
    result = json.loads(
        resolve_model_json(role, directory=_configured_model_directory(), override_path=override)
    )
    if result["downloaded"]:
        logger.info("Downloaded model {}: {}", result["model_id"], result["path"])
    return Path(result["path"]), result["sha256"]


class LazySession:
    def __init__(self, role, override=None, threads=1):
        self.role, self.override, self.threads = role, override, threads
        self.model_id = role
        self.device = "cpu"
        self._session = None
        self.cpu_spinning = None
        self.cpu_dynamic_block_base = None
        self.cpu_force_spinning_stop = None
        self.cpu_shared_arena = None
        self.partition_table = True
        self.optimize = True

    def descriptor(self):
        if self.device == "cuda":
            prepare_cuda_runtime()
        path, sha = resolve_model(self.model_id, self.override)
        enabled = self.optimize and (self.role != "table" or self.partition_table)
        optimization = native_optimization(self.role, sha, self.device, enabled)
        libraries = (
            [str(operator_library())]
            if optimization in ("ocr_detection", "ocr_recognition", "unet", "formula")
            else []
        )
        logger.info(
            "{} model {}: {} (optimization={}, intra_op_threads={})",
            self.device.upper(),
            self.role,
            path,
            optimization or "none",
            self.threads or "auto",
        )
        return json.dumps(
            dict(
                runtime=str(runtime_path()),
                model=str(path),
                sha256=sha,
                device=self.device,
                threads=self.threads,
                max_elements=sys.maxsize // 4,
                cpu_spinning=self.cpu_spinning,
                cpu_dynamic_block_base=self.cpu_dynamic_block_base,
                cpu_force_spinning_stop=self.cpu_force_spinning_stop,
                cpu_shared_arena=self.cpu_shared_arena,
                operator_libraries=libraries,
                optimization=optimization,
            )
        )

    def session(self):
        # Raw APIs keep the original complete-model input/output contract.
        from . import OnnxSession

        with _LOCK:
            if self._session is None:
                if self.device == "cuda":
                    prepare_cuda_runtime()
                runtime = runtime_path()
                path, sha = resolve_model(self.model_id, self.override)
                key = (
                    str(runtime),
                    str(path),
                    sha,
                    self.device,
                    self.threads,
                    self.cpu_spinning,
                    self.cpu_dynamic_block_base,
                    self.cpu_force_spinning_stop,
                    self.cpu_shared_arena,
                )
                if key not in _SESSIONS:
                    _SESSIONS[key] = OnnxSession(
                        runtime,
                        path,
                        sha,
                        device=self.device,
                        threads=self.threads,
                        max_elements=sys.maxsize // 4,
                        cpu_spinning=self.cpu_spinning,
                        cpu_dynamic_block_base=self.cpu_dynamic_block_base,
                        cpu_force_spinning_stop=self.cpu_force_spinning_stop,
                        cpu_shared_arena=self.cpu_shared_arena,
                    )
                self._session = _SESSIONS[key]
            return self._session

    def __getattr__(self, name):
        return getattr(self.session(), name)


class LayoutSession(LazySession):
    def __init__(self, override=None, threads=1, shape="auto"):
        super().__init__("layout", override, threads)
        self.shape = shape

    def layout_png(self, data, *, threshold=0.5):
        if self.shape == "rect" and self.model_id == "layout":
            return self.session().layout_png(data, threshold=threshold)
        result = json.loads(
            self.session()._native.layout_polygons_png_json(
                data, threshold, self.shape, geos_runtime()
            )
        )
        for i, value in enumerate(result):
            if self.model_id != "pp_doclayout_plus_l":
                value["order"] = i
        return result


def geos_runtime():
    from ._native import locate_runtime

    return str(locate_runtime(resource_hints(include_ort=False), "geos"))


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Unsupported config value: {type(value)}")


def model_config(layout_config, ocr_config, formula_config, table_config, **options):
    return json.dumps(
        dict(
            layout_config=layout_config,
            ocr_config=ocr_config,
            formula_config=formula_config,
            table_config=table_config,
            device=os.environ.get("MINERU_DEVICE_MODE", "cpu"),
            **options,
        ),
        default=_json_value,
        allow_nan=False,
    )


def load_models(
    layout_config,
    ocr_config,
    formula_config,
    table_config,
    *,
    cpu_spinning=None,
    cpu_dynamic_block_base=None,
):
    # Raw/development sessions use the same validated selection plan as production.
    from ._native import model_plan_json

    plan = json.loads(
        model_plan_json(
            model_config(
                layout_config,
                ocr_config,
                formula_config,
                table_config,
                cpu_spinning=cpu_spinning,
                cpu_dynamic_block_base=cpu_dynamic_block_base,
            )
        )
    )
    models = {}
    for name, spec in plan["models"].items():
        model = (
            LayoutSession(spec["override"], spec["threads"], plan["layout_shape"])
            if name == "layout_model"
            else LazySession(spec["role"], spec["override"], spec["threads"])
        )
        model.__dict__.update(spec)
        models[name] = model
    models["table_variant"] = plan["table_variant"]
    models["fallback_table_variant"] = plan["fallback_table_variant"]
    return models


def resource_hints(pdfium=False, include_ort=True):
    # Package layout and Python's environment are host facts. Library selection
    # and all model/optimization policy are implemented by the shared Rust SDK.
    roots = [str(Path(__file__).parent)]
    spec = importlib.util.find_spec("onnxruntime") if include_ort else None
    if spec and spec.origin:
        roots.insert(0, str(Path(spec.origin).parent / "capi"))
    if os.environ.get("RAPIDDOC_RUNTIME_DIR"):
        roots.insert(0, os.environ["RAPIDDOC_RUNTIME_DIR"])
    hints = dict(
        bundle_dirs=roots,
        ort_library=os.environ.get("RAPIDDOC_ONNXRUNTIME_LIBRARY") or None,
        geos_library=os.environ.get("RAPIDDOC_GEOS_LIBRARY") or None,
        operator_library=os.environ.get("RAPIDDOC_OPERATOR_LIBRARY")
        or os.environ.get("ONNXRUNTIME_OPS_LIBRARY")
        or None,
        model_directory=_configured_model_directory(),
        pdfium_library=None,
    )
    if pdfium:
        from ._pdfium_runtime import resolve_pdfium_library

        hints["pdfium_library"] = resolve_pdfium_library(None)
    return json.dumps(hints)


def prepare_device(device):
    if device == "cuda":
        prepare_cuda_runtime()


def log_model_event(event):
    if event.get("stage") != "model_resolved":
        return
    model = json.loads(event["detail"])
    if model["downloaded"]:
        logger.info("Downloaded model {}: {}", model["role"], model["model"])
    logger.info(
        "{} model {}: {} (optimization={}, intra_op_threads={})",
        model["device"].upper(),
        model["role"],
        model["model"],
        model["optimization"] or "none",
        model["threads"] or "auto",
    )
