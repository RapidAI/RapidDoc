"""Small production PDF input/text API; no preview pipeline imports."""

import json
import os
import sys
from dataclasses import dataclass
from ._host_context import host_call as _host_call
from ._native import parse_pdf_text_json


@dataclass
class PdfTextOutput:
    """原生文字和完整页面事实；页号保留原始索引，坐标为 PDF 原生 point。"""

    text: str
    document: dict

    @property
    def pages(self):
        """返回原始页码对应的页面事实。"""
        return self.document["pages"]

    @property
    def diagnostics(self):
        """返回未执行 OCR 等明确诊断。"""
        return self.document["diagnostics"]


def _read_bytes(source, max_input_bytes):
    """读取完整输入；仅在调用方显式指定时限制字节数。"""
    if max_input_bytes is not None and (type(max_input_bytes) is not int or max_input_bytes < 0):
        raise ValueError("max_input_bytes must be None or a non-negative integer")
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as stream:
            data = stream.read() if max_input_bytes is None else stream.read(max_input_bytes + 1)
    elif hasattr(source, "read"):
        data = source.read() if max_input_bytes is None else source.read(max_input_bytes + 1)
    else:
        raise TypeError("source must be a path, bytes, or binary stream")
    if not isinstance(data, bytes):
        raise TypeError("source stream must return bytes")
    return data


def _prepare_pdf(source, pdfium_library, max_input_bytes):
    """统一检查宿主边界、受限读取输入并定位运行库；不关闭调用方的流。"""
    data = _read_bytes(source, max_input_bytes)
    from ._pdfium_runtime import resolve_pdfium_library

    library = resolve_pdfium_library(pdfium_library)
    return str(library), data


@_host_call
def parse_pdf_text(
    source,
    *,
    pdfium_library=None,
    start_page=0,
    end_page=None,
    password=None,
    allow_partial=False,
    max_input_bytes=None,
    max_pages=None,
    max_chars_per_page=None,
    max_total_chars=None,
):
    """从路径、bytes 或二进制流提取原生 PDF 文字。

    页范围从零开始，end_page 不包含在结果中。默认拒绝需 OCR 的页；
    allow_partial=True 才允许返回部分文字及诊断。不会关闭调用方提供的流。
    使用独立 PDFium 实例，可与 pypdfium2 同进程共存；不共享双方的文档句柄。
    """
    library, data = _prepare_pdf(source, pdfium_library, max_input_bytes)
    limits = [
        sys.maxsize if value is None else value
        for value in (max_input_bytes, max_pages, max_chars_per_page, max_total_chars)
    ]
    output = json.loads(
        parse_pdf_text_json(
            str(library), data, start_page, end_page, password, allow_partial, *limits
        )
    )
    return PdfTextOutput(**output)
