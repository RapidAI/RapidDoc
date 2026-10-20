"""原生视觉块折叠、方向归一化与页面裁图。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from ...assets import image_size as _normalize_page_size
from ...foundation._coordinates import convert_bbox
from ...foundation._coordinates import normalize_bbox as _normalize_model_bbox_for_containment
from ...foundation._coordinates import normalize_quarter_turn_angle as _normalize_visual_block_angle
from ...foundation._geometry import calculate_overlap_area_in_bbox1_area_ratio
from ...foundation._image_operations import encode_crop_as_jpeg_data_uri as _encode_page_crop_as_jpeg_data_uri
from ...schema import BBox, BlockType
from .constants import (
    IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
    MODEL_JSON_VISUAL_BLOCK_TYPES,
)

if TYPE_CHECKING:
    from ._document import PDFDocument


def _attach_visual_block_images(
    model_list: list[list[dict[str, Any]]],
    images_list: list[dict[str, Any]],
    page_start_index: int = 0,
) -> None:
    """在窗口页图释放前，为最终 model_list 视觉块写入回正后的页面裁图。"""
    if len(model_list) != len(images_list):
        raise ValueError(f"Hybrid visual crop page count mismatch: model_list={len(model_list)}, images={len(images_list)}")

    for page_offset, (page_model_list, image_dict) in enumerate(zip(model_list, images_list)):
        _attach_prepared_visual_block_images(
            [_prepare_page_visual_blocks(page_model_list)], [image_dict], page_start_index + page_offset
        )


def _prepare_page_visual_blocks(page_model_list: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """先折叠视觉容器，再记录原始块索引，供按需页图任务复用。"""
    _collapse_image_blocks(page_model_list)
    return [(index, block) for index, block in enumerate(page_model_list) if block.get("type") in MODEL_JSON_VISUAL_BLOCK_TYPES]


def _visual_page_ranges(
    prepared_pages: list[list[tuple[int, dict[str, Any]]]],
    image_bytes_by_page: dict[int, int] | None = None,
    *,
    window_size: int = 64,
) -> list[tuple[int, int]]:
    """在指定窗口和 32MiB 像素预算内合并需裁图页，不改变单页清晰度。"""
    ranges: list[tuple[int, int]] = []
    batch_bytes = 0
    for page_index, blocks in enumerate(prepared_pages):
        if not blocks:
            continue
        page_bytes = (image_bytes_by_page or {}).get(page_index, 0)
        if (
            ranges
            and page_index == ranges[-1][1] + 1
            and page_index // window_size == ranges[-1][0] // window_size
            and batch_bytes + page_bytes <= 32 * 1024 * 1024
        ):
            ranges[-1] = (ranges[-1][0], page_index)
            batch_bytes += page_bytes
        else:
            ranges.append((page_index, page_index))
            batch_bytes = page_bytes
    return ranges


def attach_visual_block_images_from_pdf(
    document: PDFDocument,
    model_list: list[list[dict[str, Any]]],
    *,
    window_size: int = 64,
    timeout: int | None = None,
    threads: int | None = None,
    session=None,
) -> None:
    """按当前 PDF 全部物理页的视觉块需求原地补图；页图由本函数释放，文档仍归调用方。"""
    from .images import _load_visual_crops_from_pdf_bytes_range, get_pdf_render_backend
    from .raster import estimate_page_image_bytes

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if len(model_list) != document.page_count:
        raise ValueError(f"PDF visual crop page count mismatch: model_list={len(model_list)}, document={document.page_count}")

    prepared_visuals = [_prepare_page_visual_blocks(page) for page in model_list]
    image_bytes = {
        index: estimate_page_image_bytes(document.page_size(index)) for index, blocks in enumerate(prepared_visuals) if blocks
    }
    ranges = _visual_page_ranges(prepared_visuals, image_bytes, window_size=window_size)
    if ranges and session is None and get_pdf_render_backend() == "session":
        session = document.get_render_session(threads=threads, timeout=timeout)
    for start, end in ranges:
        crop_specs = [
            [(index, {key: block.get(key) for key in ("bbox", "angle", "type")}) for index, block in page]
            for page in prepared_visuals[start : end + 1]
        ]
        crops = _load_visual_crops_from_pdf_bytes_range(
            document.bytes,
            crop_specs,
            start_page_id=start,
            end_page_id=end,
            timeout=timeout,
            threads=threads,
            **({"session": session} if session is not None else {}),
        )
        if len(crops) != end - start + 1:
            raise ValueError("PDF visual crop page count mismatch")
        for offset, page_crops in enumerate(crops, start=start):
            for block_index, payload in page_crops:
                if payload is not None:
                    model_list[offset][block_index]["image_base64"] = payload


def _attach_prepared_visual_block_images(
    prepared_pages: list[list[tuple[int, dict[str, Any]]]],
    images_list: list[dict[str, Any]],
    page_start_index: int = 0,
) -> None:
    """按所选 PDF 的物理页索引裁图，输入块已经整理且无需再次折叠。"""
    if len(prepared_pages) != len(images_list):
        raise ValueError(f"Hybrid visual crop page count mismatch: model_list={len(prepared_pages)}, images={len(images_list)}")
    for page_offset, (visual_blocks, image_dict) in enumerate(zip(prepared_pages, images_list)):
        if not visual_blocks:
            continue

        page_index = page_start_index + page_offset
        page_pil_image = image_dict.get("img_pil")
        if page_pil_image is None:
            logger.warning(f"Skipping model visual block crops without page image: page={page_index}")
            continue

        converted_page_image = None
        try:
            if getattr(page_pil_image, "mode", None) == "RGB":
                page_rgb_image = page_pil_image
            else:
                converted_page_image = page_pil_image.convert("RGB")
                page_rgb_image = converted_page_image

            page_size = _normalize_page_size(page_rgb_image)
            np_image = np.asarray(page_rgb_image)
            for block_idx, block in visual_blocks:
                try:
                    pixel_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
                    if pixel_bbox is None:
                        raise ValueError("invalid bbox")
                    angle = _normalize_visual_block_angle(block.get("angle", 0))
                    image_base64 = _encode_page_crop_as_jpeg_data_uri(
                        np_image,
                        pixel_bbox,
                        angle,
                    )
                    if not image_base64:
                        raise ValueError("empty crop or JPEG encoding failure")
                    block["image_base64"] = image_base64
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid model visual block crop: "
                        f"page={page_index}, block={block_idx}, type={block.get('type')}, "
                        f"bbox={block.get('bbox')}, error={exc}"
                    )
        finally:
            if converted_page_image is not None:
                converted_page_image.close()


attach_visual_block_images = _attach_visual_block_images

__all__ = [
    "attach_visual_block_images",
    "attach_visual_block_images_from_pdf",
]


def _bbox_to_pixel_bbox(bbox: BBox | None, page_size: tuple[int, int]) -> BBox | None:
    """在 DocVortex 原生 ModelJson 边界解释归一化或像素框，再调用共享的显式坐标转换。"""
    if bbox is None or len(bbox) != 4:
        return None
    try:
        coordinates = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    space = "unit" if all(0.0 <= value <= 1.0 for value in coordinates) else "pixel"
    return convert_bbox(coordinates, source_space=space, target_space="pixel", page_size=page_size)


def _collapse_image_blocks(
    page_model_list: list[dict[str, Any]],
    containment_threshold: float = IMAGE_BLOCK_CONTAINMENT_THRESHOLD,
) -> None:
    """将 image_block 折叠为单个图片，并删除面积上被其包裹的非容器子块。"""
    image_blocks = [block for block in page_model_list if block.get("type") == "image_block"]
    if not image_blocks:
        return

    image_block_ids = {id(block) for block in image_blocks}
    image_block_bboxes = [
        bbox for block in image_blocks if (bbox := _normalize_model_bbox_for_containment(block.get("bbox"))) is not None
    ]

    retained_blocks: list[dict[str, Any]] = []
    for block in page_model_list:
        if id(block) in image_block_ids:
            block["type"] = BlockType.IMAGE
            retained_blocks.append(block)
            continue

        block_bbox = _normalize_model_bbox_for_containment(block.get("bbox"))
        is_contained = block_bbox is not None and any(
            calculate_overlap_area_in_bbox1_area_ratio(block_bbox, image_block_bbox) >= containment_threshold
            for image_block_bbox in image_block_bboxes
        )
        if not is_contained:
            retained_blocks.append(block)

    page_model_list[:] = retained_blocks


def _attach_owned_bitmap_crops(visual_blocks, bitmap, native, page_index):
    """直接裁剪独立位图，复用原框归一化与 JPEG 编码；原生计算错误明确传播。"""
    import base64

    from ...foundation._geometry import normalize_to_int_bbox
    from ...foundation._image_operations import _encode_rgb_as_jpeg_bytes

    data, width, height, stride, mode = bitmap
    for block_idx, block in visual_blocks:
        try:
            pixel_bbox = _bbox_to_pixel_bbox(block.get("bbox"), (width, height))
            bbox = normalize_to_int_bbox(pixel_bbox, image_size=(height, width))
            if bbox is None:
                raise ValueError("invalid bbox")
            angle = _normalize_visual_block_angle(block.get("angle", 0))
        except Exception as exc:
            logger.warning(f"Skipping invalid model visual block crop: page={page_index}, block={block_idx}, error={exc}")
            continue
        # 计算异常不能被无效输入的兼容跳过逻辑吞掉，也不能静默切回 Python。
        pixels, crop_width, crop_height = native.crop_bitmap_bgr(data, width, height, stride, mode, bbox, angle)
        crop_bgr = np.frombuffer(pixels, dtype=np.uint8).reshape(crop_height, crop_width, 3)
        try:
            # 保留原生 BGR 裁剪协议，只在 Pillow 编码边界交换颜色通道。
            encoded = _encode_rgb_as_jpeg_bytes(crop_bgr[..., ::-1])
            block["image_base64"] = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"
        except Exception as exc:
            logger.warning(f"Skipping invalid model visual block crop: page={page_index}, block={block_idx}, error={exc}")
