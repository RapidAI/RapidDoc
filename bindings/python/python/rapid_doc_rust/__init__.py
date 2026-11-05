"""Native document SDK. Historical migration APIs live in development tools."""

from importlib import import_module
from typing import TYPE_CHECKING
from ._native import VERSION as __version__, RapidDocError, resolve_page_range, validate_rect
from ._native import legacy_pipeline_json, merge_text_lines, repair_formula_latex, fix_formula_text
from ._document import PdfTextOutput, _read_bytes, _prepare_pdf, parse_pdf_text

__all__ = [
    "__version__",
    "RapidDocError",
    "RapidDoc",
    "RapidDocOutput",
    "PdfTextOutput",
    "parse_pdf_text",
    "resolve_page_range",
    "validate_rect",
    "legacy_pipeline_json",
    "merge_text_lines",
    "repair_formula_latex",
    "fix_formula_text",
    "OcrPageOutput",
    "OnnxSession",
    "RecoveredTable",
    "RenderedPage",
    "capabilities",
    "match_table",
    "match_wired_table",
    "ocr_pdf_page",
    "ocr_png",
    "recognize_table_png",
    "recognize_wired_table_png",
    "recover_pdf_table",
    "render_model_pages",
    "render_pdf_page",
]

_DIAGNOSTICS = [
    "OcrPageOutput",
    "OnnxSession",
    "RecoveredTable",
    "RenderedPage",
    "capabilities",
    "match_table",
    "match_wired_table",
    "ocr_pdf_page",
    "ocr_png",
    "recognize_table_png",
    "recognize_wired_table_png",
    "recover_pdf_table",
    "render_model_pages",
    "render_pdf_page",
]

if TYPE_CHECKING:
    from ._compat import RapidDoc as RapidDoc, RapidDocOutput as RapidDocOutput


def __getattr__(name):
    if name in ("RapidDoc", "RapidDocOutput"):
        module = import_module("._compat", __name__)
    elif name in _DIAGNOSTICS:
        module = import_module("._diagnostics", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
