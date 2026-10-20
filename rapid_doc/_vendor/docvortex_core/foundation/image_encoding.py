"""Flash 各格式复用的轻量图片编码能力。"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from PIL import Image


ImageFormat: TypeAlias = Literal["jpeg", "png", "webp"]
_IMAGE_FORMATS: dict[ImageFormat, tuple[str, str, str]] = {
    "jpeg": ("JPEG", "image/jpeg", "jpg"),
    "png": ("PNG", "image/png", "png"),
    "webp": ("WEBP", "image/webp", "webp"),
}


@dataclass(frozen=True, slots=True)
class ImageArtifact:
    """持有独立编码字节和尺寸，不依赖已关闭的图像或源文档。"""

    data: bytes
    image_format: ImageFormat
    width: int
    height: int

    def __post_init__(self) -> None:
        """拒绝未支持的格式，保证派生 MIME 与扩展名一致。"""
        if self.image_format not in _IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {self.image_format}")

    @property
    def mime_type(self) -> str:
        """返回编码格式对应的标准 MIME 类型。"""
        return _IMAGE_FORMATS[self.image_format][1]

    @property
    def extension(self) -> str:
        """返回不带点的扩展名，JPEG 统一使用 jpg。"""
        return _IMAGE_FORMATS[self.image_format][2]


def encode_image(image: Image.Image, *, image_format: ImageFormat = "jpeg") -> ImageArtifact:
    """编码调用者持有的图像，仅关闭本函数创建的颜色转换副本。"""
    if image_format not in _IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format}")
    output_image = image.convert("RGB") if image_format == "jpeg" and image.mode != "RGB" else image
    try:
        data = image_to_bytes(output_image, _IMAGE_FORMATS[image_format][0])
        return ImageArtifact(data=data, image_format=image_format, width=image.width, height=image.height)
    finally:
        if output_image is not image:
            output_image.close()


def transcode_image(data: bytes, *, image_format: ImageFormat = "jpeg") -> ImageArtifact:
    """解码内存图片并转为目标格式，在成功或失败时均释放解码图像。"""
    from PIL import Image

    with BytesIO(data) as buffer:
        image = Image.open(buffer)
        try:
            return encode_image(image, image_format=image_format)
        finally:
            image.close()


def image_to_bytes(
    image: Image.Image,
    image_format: str = "JPEG",
) -> bytes:
    """按指定格式把 Pillow 图片编码为字节。"""
    with BytesIO() as image_buffer:
        image.save(image_buffer, format=image_format)
        return image_buffer.getvalue()


def image_to_b64str(
    image: Image.Image,
    image_format: str = "JPEG",
) -> str:
    """按指定格式把 Pillow 图片编码为 data URI。"""
    image_bytes = image_to_bytes(image, image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


__all__ = ["ImageArtifact", "ImageFormat", "encode_image", "transcode_image", "image_to_b64str", "image_to_bytes"]
