"""PDF 文档访问、显式分类及共享文本契约。"""

from ._document import PDFDocument, PDFPage, PDFPageTextGeometry, PDFPageVectorGeometry, get_lines_from_chars
from .pdfium import PdfiumFontError, PdfiumRuntimeInfo, initialize_pdfium_runtime
from .text import Bbox, Char, Line, Span
from .snapshot import PDFPageSnapshot

__all__ = [
    "PDFDocument",
    "PDFRenderSession",
    "PDFPage",
    "PDFPageSnapshot",
    "PDFPageTextGeometry",
    "PDFPageVectorGeometry",
    "get_lines_from_chars",
    "Bbox",
    "Char",
    "Line",
    "Span",
    "initialize_pdfium_runtime",
    "PdfiumRuntimeInfo",
    "PdfiumFontError",
]


def __getattr__(name: str):
    """仅显式请求渲染会话时加载进程编排模块，保持文档导入边界。"""
    if name == "PDFRenderSession":
        from .render_session import PDFRenderSession

        return PDFRenderSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
