"""按需加载的图像统计与素材生成入口。"""

from __future__ import annotations

from typing import Any, Protocol

from ..schema import BBox


class ImageArray(Protocol):
    """图像数组的轻量形状契约，导入素材 SDK 时无需加载 NumPy。"""

    @property
    def shape(self) -> tuple[int, ...]:
        """返回数组各维长度，前两维表示图像高度和宽度。"""
        ...


def calculate_contrast(img: ImageArray, img_mode: str) -> float:
    """按原有 RGB/BGR 通道规则计算图像对比度。"""
    from ..foundation._image import calculate_contrast as calculate

    return calculate(img, img_mode)


def crop_pil_image(bbox: BBox, image: Any) -> Any:
    """按归一化区域裁剪 Pillow 图像，保留空框处理语义。"""
    from ..foundation._image import crop_pil_image as crop

    return crop(bbox, image)


def image_size(image: Any) -> tuple[int, int]:
    """读取 Pillow 或 NumPy 图像的宽高，不触发 PDF 运行时。"""
    if hasattr(image, "shape"):
        height, width = image.shape[:2]
        return width, height
    return image.size


def rotate_image_to_upright(image: ImageArray, angle: int) -> ImageArray:
    """按视觉块方向旋转图像，保留原有角度约定。"""
    from ..foundation._image_operations import rotate_image_to_upright as rotate

    return rotate(image, angle)


def encode_crop_as_jpeg_data_uri(image: ImageArray, bbox: BBox, angle: int) -> str:
    """按像素区域裁剪、旋转并编码 JPEG 素材。"""
    from ..foundation._image_operations import encode_crop_as_jpeg_data_uri as encode

    return encode(image, bbox, angle)


__all__ = ["calculate_contrast", "crop_pil_image", "image_size", "rotate_image_to_upright", "encode_crop_as_jpeg_data_uri"]
