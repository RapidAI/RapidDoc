"""PDF 链接与签名注解提取，保持原生提取算法与资源语义。"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from ...foundation._hyperlink import sanitize_hyperlink_target
from ...schema import BBox
from .native_contracts import _PDF_EXTERNAL_LINK_SCHEMES, PDFLinkAnnotation
from .native_coordinates import _drawing_page_size, _normalize_pdf_page_bbox, _transform_drawing_point

logger = logging.getLogger("rapid_doc._vendor.docvortex_core.document.pdf._document")


def _validate_pdf_external_link_target(value: str) -> str | None:
    """校验 PDF URI Action，只保留显式且可安全输出的外部链接。"""
    return sanitize_hyperlink_target(value, allowed_schemes=_PDF_EXTERNAL_LINK_SCHEMES)


def _get_pdfium_uri_path(raw_doc: Any, raw_action: Any) -> str | None:
    """按 PDFium 两阶段缓冲区协议读取 URI Action 的 UTF-8 目标。"""

    try:
        required = int(
            pdfium_c.FPDFAction_GetURIPath(
                raw_doc,
                raw_action,
                None,
                0,
            )
        )
    except Exception:
        return None
    if required <= 1:
        return None
    try:
        buffer = ctypes.create_string_buffer(required)
        actual = int(
            pdfium_c.FPDFAction_GetURIPath(
                raw_doc,
                raw_action,
                buffer,
                required,
            )
        )
    except Exception:
        return None
    if actual <= 1 or actual > required:
        return None
    raw_value = bytes(buffer.raw[: actual - 1])
    try:
        return raw_value.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _pdf_link_annotation_is_visible(page: pdfium.PdfPage, raw_link: Any) -> bool:
    """读取 Link 注解可见性；损坏或显式隐藏的注解均不参与文本富化。"""

    raw_annot = None
    try:
        raw_annot = pdfium_c.FPDFLink_GetAnnot(page.raw, raw_link)
        if not raw_annot:
            return False
        flags = int(pdfium_c.FPDFAnnot_GetFlags(raw_annot))
    except Exception:
        return False
    finally:
        if raw_annot:
            try:
                pdfium_c.FPDFPage_CloseAnnot(raw_annot)
            except Exception:
                pass
    hidden_flags = pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE | pdfium_c.FPDF_ANNOT_FLAG_HIDDEN | pdfium_c.FPDF_ANNOT_FLAG_NOVIEW
    return not bool(flags & hidden_flags)


def _visual_bbox_from_pdf_points(
    points: list[tuple[float, float]],
    page_bbox: BBox,
    page_rotation: int,
) -> BBox | None:
    """把 PDF 底左坐标点集转换并裁剪为视觉页面左上坐标 bbox。"""

    if not points or not all(math.isfinite(coordinate) for point in points for coordinate in point):
        return None
    visual_points = [_transform_drawing_point(point, page_bbox, page_rotation) for point in points]
    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    visual_bbox = (
        max(0.0, min(point[0] for point in visual_points)),
        max(0.0, min(point[1] for point in visual_points)),
        min(page_width, max(point[0] for point in visual_points)),
        min(page_height, max(point[1] for point in visual_points)),
    )
    if visual_bbox[2] <= visual_bbox[0] or visual_bbox[3] <= visual_bbox[1]:
        return None
    return visual_bbox


def _pdf_link_region_bboxes(
    raw_link: Any,
    page_bbox: BBox,
    page_rotation: int,
) -> tuple[BBox, ...]:
    """优先读取 Link QuadPoints，并在缺失时回退到注解矩形。"""

    bboxes: list[BBox] = []
    try:
        quad_count = max(0, int(pdfium_c.FPDFLink_CountQuadPoints(raw_link)))
    except Exception:
        quad_count = 0
    for quad_index in range(quad_count):
        quad = pdfium_c.FS_QUADPOINTSF()
        try:
            ok = pdfium_c.FPDFLink_GetQuadPoints(
                raw_link,
                quad_index,
                ctypes.byref(quad),
            )
        except Exception:
            continue
        if not ok:
            continue
        bbox = _visual_bbox_from_pdf_points(
            [
                (float(quad.x1), float(quad.y1)),
                (float(quad.x2), float(quad.y2)),
                (float(quad.x3), float(quad.y3)),
                (float(quad.x4), float(quad.y4)),
            ],
            page_bbox,
            page_rotation,
        )
        if bbox is not None and bbox not in bboxes:
            bboxes.append(bbox)

    if not bboxes:
        rect = pdfium_c.FS_RECTF()
        try:
            ok = pdfium_c.FPDFLink_GetAnnotRect(raw_link, ctypes.byref(rect))
        except Exception:
            ok = False
        if ok:
            bbox = _visual_bbox_from_pdf_points(
                [
                    (float(rect.left), float(rect.bottom)),
                    (float(rect.left), float(rect.top)),
                    (float(rect.right), float(rect.bottom)),
                    (float(rect.right), float(rect.top)),
                ],
                page_bbox,
                page_rotation,
            )
            if bbox is not None:
                bboxes.append(bbox)
    return tuple(
        sorted(
            bboxes,
            key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]),
        )
    )


def _extract_page_link_annotations(
    page: pdfium.PdfPage,
    raw_doc: Any,
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFLinkAnnotation]:
    """枚举页面外部 URI Link；单条损坏数据不会中断同页其他链接。"""

    annotations: list[PDFLinkAnnotation] = []
    position = ctypes.c_int(0)
    source_index = 0
    while True:
        raw_link = pdfium_c.FPDF_LINK()
        previous_position = int(position.value)
        try:
            found = pdfium_c.FPDFLink_Enumerate(
                page.raw,
                ctypes.byref(position),
                ctypes.byref(raw_link),
            )
        except Exception:
            break
        if not found:
            break
        current_source_index = source_index
        source_index += 1
        if int(position.value) <= previous_position:
            break
        try:
            if not raw_link or not _pdf_link_annotation_is_visible(page, raw_link):
                continue
            raw_action = pdfium_c.FPDFLink_GetAction(raw_link)
            if not raw_action or int(pdfium_c.FPDFAction_GetType(raw_action)) != pdfium_c.PDFACTION_URI:
                continue
            raw_target = _get_pdfium_uri_path(raw_doc, raw_action)
            target = _validate_pdf_external_link_target(raw_target) if raw_target is not None else None
            if target is None:
                continue
            bboxes = _pdf_link_region_bboxes(
                raw_link,
                page_bbox,
                page_rotation,
            )
            if not bboxes:
                continue
            annotations.append(
                PDFLinkAnnotation(
                    target=target,
                    bboxes=bboxes,
                    source_index=current_source_index,
                )
            )
        except Exception:
            continue
    return annotations


def _get_annotation_string(raw_annot: Any, key: bytes) -> str | None:
    """读取 PDFium 注释字典中的 UTF-16 字符串或名称，异常和空值统一返回 None。"""

    try:
        required = int(pdfium_c.FPDFAnnot_GetStringValue(raw_annot, key, None, 0))
    except Exception:
        return None
    if required <= 2:
        return None
    try:
        buffer = (ctypes.c_ushort * ((required + 1) // 2))()
        actual = int(
            pdfium_c.FPDFAnnot_GetStringValue(
                raw_annot,
                key,
                buffer,
                ctypes.sizeof(buffer),
            )
        )
    except Exception:
        return None
    if actual <= 2 or actual > ctypes.sizeof(buffer):
        return None
    try:
        return bytes(buffer)[: actual - 2].decode("utf-16-le")
    except UnicodeDecodeError:
        return None


def _signature_bbox_from_annotation(
    raw_annot: Any,
    page_bbox: BBox,
    page_rotation: int,
    form_handle: Any | None,
) -> BBox | None:
    """校验一个可见签名 Widget，并把注释矩形裁剪到视觉页面坐标。"""

    try:
        if int(pdfium_c.FPDFAnnot_GetSubtype(raw_annot)) != pdfium_c.FPDF_ANNOT_WIDGET:
            return None
        flags = int(pdfium_c.FPDFAnnot_GetFlags(raw_annot))
    except Exception:
        return None
    hidden_flags = pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE | pdfium_c.FPDF_ANNOT_FLAG_HIDDEN | pdfium_c.FPDF_ANNOT_FLAG_NOVIEW
    if flags & hidden_flags:
        return None
    field_type: int | None = None
    if form_handle is not None:
        try:
            field_type = int(pdfium_c.FPDFAnnot_GetFormFieldType(form_handle, raw_annot))
        except Exception:
            field_type = None
    if field_type == pdfium_c.FPDF_FORMFIELD_SIGNATURE:
        pass
    elif field_type is None or field_type <= pdfium_c.FPDF_FORMFIELD_UNKNOWN:
        if _get_annotation_string(raw_annot, b"FT") != "Sig":
            return None
    else:
        return None

    # 正常外观长度只有 UTF-16 终止符时等价于无 /AP /N，不能形成可见签名。
    try:
        appearance_length = int(
            pdfium_c.FPDFAnnot_GetAP(
                raw_annot,
                pdfium_c.FPDF_ANNOT_APPEARANCEMODE_NORMAL,
                None,
                0,
            )
        )
    except Exception:
        return None
    if appearance_length <= 2:
        return None

    rect = pdfium_c.FS_RECTF()
    try:
        if not pdfium_c.FPDFAnnot_GetRect(raw_annot, ctypes.byref(rect)):
            return None
    except Exception:
        return None
    raw_bbox = _normalize_pdf_page_bbox((float(rect.left), float(rect.bottom), float(rect.right), float(rect.top)))
    if not all(math.isfinite(value) for value in raw_bbox):
        return None
    page_points = [
        _transform_drawing_point(point, page_bbox, page_rotation)
        for point in (
            (raw_bbox[0], raw_bbox[1]),
            (raw_bbox[0], raw_bbox[3]),
            (raw_bbox[2], raw_bbox[1]),
            (raw_bbox[2], raw_bbox[3]),
        )
    ]
    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    visual_bbox = (
        max(0.0, min(point[0] for point in page_points)),
        max(0.0, min(point[1] for point in page_points)),
        min(page_width, max(point[0] for point in page_points)),
        min(page_height, max(point[1] for point in page_points)),
    )
    if visual_bbox[2] <= visual_bbox[0] or visual_bbox[3] <= visual_bbox[1]:
        return None
    return visual_bbox


def _extract_page_signature_bboxes(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
    *,
    form_handle: Any | None = None,
) -> list[BBox]:
    """遍历页面签名注释；逐个关闭句柄，并隔离损坏注释造成的异常。"""

    try:
        annot_count = max(0, int(pdfium_c.FPDFPage_GetAnnotCount(page.raw)))
    except Exception:
        return []
    signature_bboxes: list[BBox] = []
    for annot_index in range(annot_count):
        raw_annot = None
        try:
            raw_annot = pdfium_c.FPDFPage_GetAnnot(page.raw, annot_index)
            if not raw_annot:
                continue
            signature_bbox = _signature_bbox_from_annotation(
                raw_annot,
                page_bbox,
                page_rotation,
                form_handle,
            )
            if signature_bbox is not None:
                signature_bboxes.append(signature_bbox)
        except Exception:
            # 单个损坏注释不能影响同页其他有效签名框。
            continue
        finally:
            if raw_annot:
                try:
                    pdfium_c.FPDFPage_CloseAnnot(raw_annot)
                except Exception:
                    pass
    return sorted(signature_bboxes, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))
