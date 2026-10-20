"""文档素材的显式字节所有权、图像转码与读取接口。"""

from ..foundation._image_payload import parse_image_data_uri_strict, validate_image_sidecar_path
from ..foundation.image_encoding import ImageArtifact, ImageFormat, transcode_image
from .images import (
    ImageArray,
    calculate_contrast,
    crop_pil_image,
    encode_crop_as_jpeg_data_uri,
    image_size,
    rotate_image_to_upright,
)
from .store import AssetStore

__all__ = [
    "AssetStore",
    "ImageArtifact",
    "ImageFormat",
    "ImageArray",
    "parse_image_data_uri_strict",
    "transcode_image",
    "validate_image_sidecar_path",
    "calculate_contrast",
    "crop_pil_image",
    "image_size",
    "rotate_image_to_upright",
    "encode_crop_as_jpeg_data_uri",
]
