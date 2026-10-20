"""保留原生表格单元格内的媒体，并隔离图标自己的线条与文字。"""

from __future__ import annotations

import base64
import html
import math
import re
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from loguru import logger

from ....foundation.image_encoding import encode_image
from ....schema import BBox
from .geometry import _bbox_area, _bbox_overlap_in_first, _coerce_bbox, _expand_bbox
from ._table_recovery.geometry import page_bbox_to_table_local

if TYPE_CHECKING:
    from ..contracts import NativePdfSource
    from .models import _PageSource
    from ._table_recovery.contracts import NativeTableInput, NativeTableResult


def nested_table_media(source: _PageSource, table_bbox: BBox) -> list[BBox]:
    """只选择完整位于父表内的小媒体；包含网格的大图继续交给图形规则。"""
    area = _bbox_area(table_bbox)
    forms = []
    for box in source.form_bboxes:
        paths = [path for path in source.path_infos
                 if (path.fill_visible or path.stroke_visible)
                 and path.segment_count >= 4 and _bbox_overlap_in_first(path.bbox, box) >= 0.99]
        # 只封装文字的 Form 继续输出原生文字，单个底色矩形也不是图标。
        if len(paths) >= 2 or any(path.segment_count >= 6 for path in paths):
            forms.append(box)
    boxes = [
        box for box in [*forms, *source.image_bboxes]
        if 0 < _bbox_area(box) <= 0.25 * area
        and _bbox_overlap_in_first(box, table_bbox) >= 0.99
    ]
    # 嵌套 Form 只截取最外层可见图标，不重复输出其子图。
    result = []
    for box in sorted(boxes, key=_bbox_area, reverse=True):
        if not any(_bbox_overlap_in_first(box, parent) >= 0.99 for parent in result):
            result.append(box)
    return sorted(result, key=lambda box: (box[1], box[0]))


def mask_table_media(table_input: NativeTableInput, boxes: list[BBox]) -> NativeTableInput:
    """网格保持页面坐标；仅移除完全落在媒体内部的原语和字符。"""
    masks = [_expand_bbox(box, 0.5) for box in boxes]

    def inside(bounds):
        box = _coerce_bbox(bounds)
        return box is not None and any(_bbox_overlap_in_first(box, mask) >= 0.99 for mask in masks)

    return replace(
        table_input,
        drawing_lines=tuple(rule for rule in table_input.drawing_lines if not inside(rule.bbox)),
        rectangles=tuple(rect for rect in table_input.rectangles if not inside(rect.bbox)),
        chars=tuple(char for char in table_input.chars if not inside(char.get("tight_bbox") or char.get("bbox"))),
    )


def attach_table_media(content: str, recovered: NativeTableResult, table_input: NativeTableInput,
                       boxes: list[BBox]) -> tuple[str, list[dict[str, Any]]] | None:
    """媒体须完整落入唯一物理单元格，失败时由调用方保留原文字恢复路径。"""
    if recovered.source != "vector_grid":
        return None
    cells = sorted(recovered.cells, key=lambda cell: (cell.row, cell.col))
    images = []
    by_cell = {}
    for box in boxes:
        local = page_bbox_to_table_local(box, table_input.table_bbox, table_input.angle)
        matches = [i for i, cell in enumerate(cells) if local is not None
                   and _bbox_overlap_in_first(local, _expand_bbox(cell.bbox, 0.5)) >= 0.99]
        if len(matches) != 1:
            return None
        # 图片编码失败时使用图内原文字，不能留下不可解析的占位 URI。
        fallback = "".join(str(char.get("char", "")) for char in table_input.chars
                           if (bounds := _coerce_bbox(char.get("tight_bbox") or char.get("bbox"))) is not None
                           and _bbox_overlap_in_first(bounds, _expand_bbox(box, 0.5)) >= 0.99)
        index = len(images)
        images.append({"bbox": box, "text": fallback})
        by_cell.setdefault(matches[0], []).append(f'<img src="docvortex-table-image:{index}" alt="Image" />')
    cell_index = 0

    def insert(match):
        nonlocal cell_index
        extra = "".join(by_cell.get(cell_index, []))
        cell_index += 1
        return match.group(1) + extra + match.group(2) + "</td>"

    markup = re.sub(r"(<td\b[^>]*>)(.*?)</td>", insert, content, flags=re.S)
    return (markup, images) if cell_index == len(cells) else None


def materialize_table_media(pages: list[list[dict[str, Any]]], pdf_doc: NativePdfSource) -> None:
    """每页只渲染一次，将临时单元格图片编码为独立 PNG。"""
    from PIL import Image

    for page_index, blocks in enumerate(pages):
        tables = [block for block in blocks if block.get("_table_images")]
        if not tables:
            continue
        page_image = None
        try:
            page_image = pdf_doc.render_page(page_index, scale=2.0).pil_image
            for table in tables:
                angle = table.get("angle", 0)
                for index, item in enumerate(table["_table_images"]):
                    box = item["bbox"]
                    crop_box = (max(0, math.floor(box[0] * page_image.width)),
                                max(0, math.floor(box[1] * page_image.height)),
                                min(page_image.width, math.ceil(box[2] * page_image.width)),
                                min(page_image.height, math.ceil(box[3] * page_image.height)))
                    crop = page_image.crop(crop_box)
                    try:
                        if angle in {90, 180, 270}:
                            rotated = crop.transpose({90: Image.Transpose.ROTATE_90,
                                                      180: Image.Transpose.ROTATE_180,
                                                      270: Image.Transpose.ROTATE_270}[angle])
                            crop.close()
                            crop = rotated
                        artifact = encode_image(crop, image_format="png")
                        uri = "data:image/png;base64," + base64.b64encode(artifact.data).decode("ascii")
                        table["content"] = table["content"].replace(f"docvortex-table-image:{index}\"", uri + '"')
                    finally:
                        crop.close()
        except Exception as exc:
            logger.warning("Native table image rendering failed on page {}: {}", page_index, exc)
        finally:
            if page_image is not None:
                page_image.close()
            for table in tables:
                for index, item in enumerate(table.pop("_table_images")):
                    table["content"] = table["content"].replace(
                        f'<img src="docvortex-table-image:{index}" alt="Image" />', html.escape(item["text"]))
