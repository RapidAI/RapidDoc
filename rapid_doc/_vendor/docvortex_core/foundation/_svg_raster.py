"""把外部 SVG 图片资源受控光栅化为 PNG data URI 的共享能力。"""

from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ElementTree
from io import BytesIO
from pathlib import PurePosixPath
from typing import Final

from loguru import logger

from ._image_payload import (
    MAX_DECODED_RASTER_DIMENSION,
    MAX_DECODED_RASTER_PIXELS,
    _parse_svg_root_strict,
)

SVG_MEDIA_TYPE: Final = "image/svg+xml"
SVG_IMAGE_EXTENSION: Final = ".svg"
# 小尺寸 logo 按固有像素直出会明显发虚，统一放大到最小可视长边。
_MIN_RENDER_LONG_EDGE: Final = 512
_SVG_LENGTH_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*(px|pt|pc|mm|cm|in|q|%)?$", re.IGNORECASE)
_LENGTH_TO_PX: Final = {
    "": 1.0,
    "px": 1.0,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "mm": 96.0 / 25.4,
    "cm": 96.0 / 2.54,
    "in": 96.0,
    "q": 96.0 / 101.6,
}
_SVG_PAYLOAD_SNIFF_WINDOW: Final = 4096


def is_svg_image_part(part_name: object | None = None, content_type: str | None = None) -> bool:
    """根据部件扩展名和声明媒体类型判断是否为 SVG 图片部件。"""
    normalized_type = (content_type or "").split(";", 1)[0].strip().casefold()
    if normalized_type == SVG_MEDIA_TYPE:
        return True
    suffix = PurePosixPath(str(part_name or "")).suffix.casefold()
    return suffix == SVG_IMAGE_EXTENSION


def looks_like_svg_payload(image_data: bytes) -> bool:
    """按载荷前缀嗅探 SVG 根节点，兜底识别被错误标注的图片部件。"""
    head = image_data[:_SVG_PAYLOAD_SNIFF_WINDOW].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<svg:"):
        return True
    return head.startswith(b"<?xml") and b"<svg" in head


def _length_to_px(value: str | None) -> float | None:
    """把 SVG width/height 长度解析为像素值，百分比等上下文相关单位返回 None。"""
    match = _SVG_LENGTH_RE.match((value or "").strip())
    if match is None:
        return None
    unit = match.group(2) or ""
    if unit == "%":
        return None
    return float(match.group(1)) * _LENGTH_TO_PX[unit.casefold()]


def _intrinsic_size(root: ElementTree.Element) -> tuple[int, int] | None:
    """从 svg 根元素解析固有像素尺寸，width/height 不可用时回退 viewBox。"""
    width = _length_to_px(root.get("width"))
    height = _length_to_px(root.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        return max(1, round(width)), max(1, round(height))
    view_box = (root.get("viewBox") or "").replace(",", " ").split()
    if len(view_box) == 4:
        try:
            box_width, box_height = float(view_box[2]), float(view_box[3])
        except ValueError:
            return None
        if box_width > 0 and box_height > 0:
            return max(1, round(box_width)), max(1, round(box_height))
    return None


def _clamped_size(width: int, height: int) -> tuple[int, int]:
    """按光栅单边与总像素预算等比收缩目标尺寸。"""
    scale = min(
        1.0,
        MAX_DECODED_RASTER_DIMENSION / width,
        MAX_DECODED_RASTER_DIMENSION / height,
        math.sqrt(MAX_DECODED_RASTER_PIXELS / (width * height)),
    )
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def _render_size(
    intrinsic: tuple[int, int] | None,
    size_hint: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """解析渲染目标尺寸：显式提示优先，其次固有尺寸并放大到最小长边。"""
    if size_hint is not None and size_hint[0] > 0 and size_hint[1] > 0:
        return _clamped_size(max(1, round(size_hint[0])), max(1, round(size_hint[1])))
    if intrinsic is None:
        return None
    width, height = intrinsic
    long_edge = max(width, height)
    if long_edge < _MIN_RENDER_LONG_EDGE:
        width = max(1, round(width * _MIN_RENDER_LONG_EDGE / long_edge))
        height = max(1, round(height * _MIN_RENDER_LONG_EDGE / long_edge))
    return _clamped_size(width, height)


def _is_fully_transparent(png: bytes) -> bool:
    """检测光栅化结果是否全透明，用于让无视觉内容的 SVG 走调用方降级。"""
    from PIL import Image

    with Image.open(BytesIO(png)) as image:
        if image.mode not in {"RGBA", "LA"}:
            return False
        return image.getextrema()[-1][1] == 0


def serialize_svg_image(image_data: bytes, *, size_hint: tuple[int, int] | None = None) -> str | None:
    """把外部 SVG 字节光栅化为透明 PNG data URI，无法安全渲染时返回 None。

    resvg 不执行脚本也不加载外部资源，是外部 SVG 进入输出的安全净化边界；
    ``skip_system_fonts=True`` 保证跨平台输出确定，代价是未转路径的 ``<text>``
    不渲染文字。DTD、实体声明和超预算载荷在解析阶段即被拒绝。
    """
    try:
        root = _parse_svg_root_strict(image_data)
    except ValueError as exc:
        logger.warning(f"SVG image payload cannot be parsed safely: {exc}")
        return None

    import resvg_py

    target_size = _render_size(_intrinsic_size(root), size_hint)
    # 严格解析后的树重序列化为 UTF-8，避免原始声明的非 UTF-8 编码干扰渲染输入。
    markup = ElementTree.tostring(root, encoding="unicode")
    render_options: dict[str, int] = {}
    if target_size is not None:
        render_options["width"] = target_size[0]
        render_options["height"] = target_size[1]
    try:
        png = resvg_py.svg_to_bytes(svg_string=markup, skip_system_fonts=True, **render_options)
        if not png:
            return None
        if _is_fully_transparent(png):
            logger.debug("Rasterized SVG image is fully transparent; degrading to caller fallback")
            return None
    except Exception as exc:  # noqa: BLE001 - resvg 异常面未定型，统一降级由调用方兜底
        logger.warning(f"SVG image cannot be rasterized: {exc}")
        return None

    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"
