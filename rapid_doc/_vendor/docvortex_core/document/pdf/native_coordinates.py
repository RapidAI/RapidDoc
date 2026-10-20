"""PDF 页面坐标、矩阵与扩展字符几何，保持原生提取算法与资源语义。"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from ...schema import BBox
from .text._contracts import Char

logger = logging.getLogger("rapid_doc._vendor.docvortex_core.document.pdf._document")


def _normalize_pdf_page_bbox(bbox: tuple[float, float, float, float]) -> BBox:
    """规范化 PDFium 页面框，兼容上下坐标顺序相反的测试或异常文档。"""
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _drawing_page_size(page_bbox: BBox, page_rotation: int) -> tuple[float, float]:
    """根据未旋转页面框和页面旋转角计算左上坐标系中的页面尺寸。"""
    width = page_bbox[2] - page_bbox[0]
    height = page_bbox[3] - page_bbox[1]
    if page_rotation in (90, 270):
        return height, width
    return width, height


def _transform_drawing_point(
    point: tuple[float, float],
    page_bbox: BBox,
    page_rotation: int,
) -> tuple[float, float]:
    """把 PDF 底左原点坐标转换为应用页面旋转后的左上原点坐标。"""
    x, y = point
    left, bottom, right, top = page_bbox
    if page_rotation == 90:
        return y - bottom, x - left
    if page_rotation == 180:
        return right - x, y - bottom
    if page_rotation == 270:
        return top - y, right - x
    return x - left, top - y


def _char_visual_bbox_from_pdfium(
    left: float,
    right: float,
    bottom: float,
    top: float,
    page_bbox: BBox,
    page_rotation: int,
) -> BBox | None:
    """把 PDFium 字符 user-space 框转换为合法视觉页面 bbox。"""
    points = [
        _transform_drawing_point(point, page_bbox, page_rotation)
        for point in (
            (left, bottom),
            (left, top),
            (right, bottom),
            (right, top),
        )
    ]
    bbox = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )
    if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _extract_page_char_extended_geometry(
    textpage: pdfium.PdfTextPage,
    chars: list[Char],
    page_bbox: BBox,
    page_rotation: int,
) -> tuple[dict[int, BBox], dict[int, BBox], dict[int, tuple[float, float]]]:
    """逐字符读取 loose/tight/origin；单字符失败不影响其余文本。"""
    loose_bboxes: dict[int, BBox] = {}
    tight_bboxes: dict[int, BBox] = {}
    origins: dict[int, tuple[float, float]] = {}
    textpage_raw = textpage.raw

    for char in chars:
        raw_char_idx = char.get("char_idx")
        if isinstance(raw_char_idx, bool) or not isinstance(raw_char_idx, int):
            continue

        try:
            char_rotation = float(char.get("rotation") or 0.0)
        except (TypeError, ValueError):
            char_rotation = math.nan
        # pdftext 已在 char["bbox"] 中保留零旋转 loose；side-map 只记录
        # 非零旋转或来源不明字符的显式覆盖，避免整本重复分配 bbox tuple。
        if not math.isfinite(char_rotation) or abs(char_rotation) > 1e-9:
            loose_rect = pdfium_c.FS_RECTF()
            try:
                has_loose_bbox = pdfium_c.FPDFText_GetLooseCharBox(
                    textpage_raw,
                    raw_char_idx,
                    loose_rect,
                )
            except Exception:
                has_loose_bbox = False
            if has_loose_bbox:
                loose_bbox = _char_visual_bbox_from_pdfium(
                    float(loose_rect.left),
                    float(loose_rect.right),
                    float(loose_rect.bottom),
                    float(loose_rect.top),
                    page_bbox,
                    page_rotation,
                )
                if loose_bbox is not None:
                    loose_bboxes[raw_char_idx] = loose_bbox

        left = ctypes.c_double()
        right = ctypes.c_double()
        bottom = ctypes.c_double()
        top = ctypes.c_double()
        try:
            has_tight_bbox = pdfium_c.FPDFText_GetCharBox(
                textpage_raw,
                raw_char_idx,
                left,
                right,
                bottom,
                top,
            )
        except Exception:
            has_tight_bbox = False
        if has_tight_bbox:
            tight_bbox = _char_visual_bbox_from_pdfium(
                left.value,
                right.value,
                bottom.value,
                top.value,
                page_bbox,
                page_rotation,
            )
            if tight_bbox is not None:
                tight_bboxes[raw_char_idx] = tight_bbox

        origin_x = ctypes.c_double()
        origin_y = ctypes.c_double()
        try:
            has_origin = pdfium_c.FPDFText_GetCharOrigin(
                textpage_raw,
                raw_char_idx,
                origin_x,
                origin_y,
            )
        except Exception:
            has_origin = False
        if has_origin:
            origin = _transform_drawing_point(
                (origin_x.value, origin_y.value),
                page_bbox,
                page_rotation,
            )
            if all(math.isfinite(value) for value in origin):
                origins[raw_char_idx] = origin

    return loose_bboxes, tight_bboxes, origins


def _multiply_pdf_matrices(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """按 PDF 行向量约定合并对象矩阵与父 Form 矩阵。"""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _apply_pdf_matrix(
    point: tuple[float, float],
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """将 PDF 仿射矩阵应用到一个路径点。"""
    x, y = point
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _get_raw_object_matrix(raw_obj: Any) -> tuple[float, float, float, float, float, float] | None:
    """读取一个原始 PDFium 页面对象矩阵，读取失败时返回 None。"""
    matrix = pdfium_c.FS_MATRIX()
    try:
        ok = pdfium_c.FPDFPageObj_GetMatrix(raw_obj, ctypes.byref(matrix))
    except Exception:
        return None
    if not ok:
        return None
    return (
        float(matrix.a),
        float(matrix.b),
        float(matrix.c),
        float(matrix.d),
        float(matrix.e),
        float(matrix.f),
    )
