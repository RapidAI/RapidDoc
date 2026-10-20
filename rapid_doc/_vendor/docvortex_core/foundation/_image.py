"""跨模型共享的轻量图像统计与裁剪原语。"""

import numpy as np
from PIL import Image

from ..schema import BBox
from ._geometry import normalize_to_int_bbox


def calculate_contrast(img: np.ndarray, img_mode: str) -> float:
    """
    计算给定图像的对比度。
    :param img: 图像，类型为numpy.ndarray
    :Param img_mode = 图像的色彩通道，'rgb' 或 'bgr'
    :return: 图像的对比度值
    """
    if img_mode not in ("rgb", "bgr"):
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] not in (3, 4) or img.size == 0:
        raise ValueError("Expected a non-empty three- or four-channel image")

    rgb = img[..., :3] if img_mode == "rgb" else img[..., 2::-1]
    if img.dtype in (np.uint8, np.uint16):
        # 保留既有 15 位定点系数与半值进位；16 位输入的最大累加值也不会超过 int32。
        values = rgb.astype(np.int32)
        gray_img = ((values[..., 0] * 9798 + values[..., 1] * 19235 + values[..., 2] * 3735 + 16384) >> 15).astype(img.dtype)
    elif img.dtype == np.float32:
        gray_img = rgb[..., 0] * np.float32(0.299) + rgb[..., 1] * np.float32(0.587) + rgb[..., 2] * np.float32(0.114)
    else:
        raise ValueError("Unsupported image depth; expected uint8, uint16 or float32")

    # 计算均值和标准差
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # 对比度定义为标准差除以平均值（加上小常数避免除零错误）
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}")
    return round(float(contrast), 2)


def crop_pil_image(bbox: BBox, image: Image.Image) -> Image.Image:
    """按 0-1 归一化 bbox 裁剪 Pillow 图像。"""
    width, height = image.size
    scaled_bbox = normalize_to_int_bbox(
        [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height],
        image_size=(height, width),
    )
    if scaled_bbox is None:
        return image.crop((0, 0, 0, 0))
    return image.crop(scaled_bbox)


__all__ = ["calculate_contrast", "crop_pil_image"]
