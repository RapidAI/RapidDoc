"""无 PDF 句柄依赖的图像裁剪、旋转及编码。"""

from __future__ import annotations

import base64
from contextlib import closing
from io import BytesIO

import numpy as np
from PIL import Image

from ..schema import BBox
from ._geometry import normalize_to_int_bbox


def rotate_image_to_upright(image: np.ndarray, angle: int) -> np.ndarray:
    """按 layout 视觉块角度把裁图旋转至正向，角度语义与方向分类模型保持一致。"""
    turns = {90: 1, 180: 2, 270: 3}.get(angle)
    if turns is None:
        return image
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        raise ValueError("Expected a non-empty image array")
    # 实际旋转返回独立连续数组，避免 NumPy 视图的负步长与原图共享写入。
    return np.rot90(image, turns).copy(order="C")


def _encode_rgb_as_jpeg_bytes(image: np.ndarray) -> bytes:
    """沿用既有 JPEG 质量与采样设置，并及时释放临时图像和内存缓冲。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] not in (3, 4) or image.size == 0:
        raise ValueError("Expected a non-empty three- or four-channel image")
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError("Unsupported image depth; expected uint8, uint16 or float32")
    rgb = image[..., :3]
    if rgb.dtype != np.uint8:
        # 高位深输入保持原编码器的饱和舍入规则，透明通道沿用既有忽略行为。
        with np.errstate(invalid="ignore"):
            rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    with closing(Image.fromarray(rgb)) as pil_image, BytesIO() as output:
        pil_image.save(output, format="JPEG", quality=95, subsampling=2, optimize=False, progressive=False)
        return output.getvalue()


def encode_crop_as_jpeg_data_uri(
    np_image: np.ndarray,
    page_bbox: BBox,
    angle: int,
) -> str:
    """从页面原图按像素框裁剪，按视觉块方向回正后编码为 JPEG data URI。"""
    image_h, image_w = np_image.shape[:2]
    image_bbox = normalize_to_int_bbox(page_bbox, image_size=(image_h, image_w))
    if image_bbox is None:
        return ""
    x0, y0, x1, y1 = image_bbox
    crop_rgb = np_image[y0:y1, x0:x1].copy()
    if crop_rgb.size == 0:
        return ""

    crop_rgb = rotate_image_to_upright(crop_rgb, angle)
    try:
        encoded = _encode_rgb_as_jpeg_bytes(crop_rgb)
    except OSError:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"
