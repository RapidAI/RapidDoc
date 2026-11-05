"""Compatibility writers over the shared Rust vector drawing implementation."""

import json
from pathlib import Path
from ._native import draw_pdf_bytes


class DataWriter:
    """Compatibility marker for caller-provided write(path, bytes) objects."""

    pass


def _draw(pdf_info, pdf_bytes, out_path, filename, kind):
    data = draw_pdf_bytes(pdf_bytes, json.dumps(pdf_info, allow_nan=False), kind)
    if isinstance(out_path, DataWriter):
        out_path.write(filename, data)
    else:
        (Path(out_path) / filename).write_bytes(data)


def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    _draw(pdf_info, pdf_bytes, out_path, filename, "layout")


def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    _draw(pdf_info, pdf_bytes, out_path, filename, "span")


def draw_line_sort_bbox(pdf_info, pdf_bytes, out_path, filename):
    _draw(pdf_info, pdf_bytes, out_path, filename, "line_sort")
