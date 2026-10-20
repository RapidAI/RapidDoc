from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfPage

from .pdfium import pdfium_guard

DEFAULT_PDF_IMAGE_DPI = 200
DEFAULT_MAX_RENDER_EDGE = 3500


def _render_page_bitmap(page: PdfPage, dpi: int, max_width_or_height: int):
    """在调用方 PDFium 锁内使用完全相同的缩放规则，返回须由调用方关闭的位图。"""
    scale = dpi / 72
    long_side_length = max(*page.get_size())
    if (long_side_length * scale) > max_width_or_height:
        scale = max_width_or_height / long_side_length
    return page.render(scale=scale), scale


def page_to_pixel_file(page: PdfPage, path, dpi: int = DEFAULT_PDF_IMAGE_DPI):
    """直接写出原始位图缓冲及解码元数据，避免 worker 构造 PIL 和完整字节副本。"""
    from pypdfium2.internal import BitmapTypeToStrReverse

    with pdfium_guard():
        bitmap = None
        try:
            bitmap, scale = _render_page_bitmap(page, dpi, DEFAULT_MAX_RENDER_EDGE)
            with open(path, "wb") as output:
                output.write(bitmap.buffer)
            return {
                "scale": scale,
                "mode": BitmapTypeToStrReverse[bitmap.format],
                "size": (bitmap.width, bitmap.height),
                "raw_mode": bitmap.mode,
                "stride": bitmap.stride,
                "path": path,
            }
        finally:
            if bitmap is not None:
                bitmap.close()


def page_to_shared_pixels(page: PdfPage, name, dpi: int = DEFAULT_PDF_IMAGE_DPI):
    """复制原始位图到调用方登记的共享内存，关闭 PDFium 句柄后由父进程解码并释放。"""
    from multiprocessing.shared_memory import SharedMemory
    from pypdfium2.internal import BitmapTypeToStrReverse

    with pdfium_guard():
        bitmap = storage = None
        try:
            bitmap, scale = _render_page_bitmap(page, dpi, DEFAULT_MAX_RENDER_EDGE)
            size = len(bitmap.buffer)
            storage = SharedMemory(name=name, create=True, size=size)
            storage.buf[:size] = memoryview(bitmap.buffer).cast("B")
            return {
                "scale": scale,
                "mode": BitmapTypeToStrReverse[bitmap.format],
                "size": (bitmap.width, bitmap.height),
                "raw_mode": bitmap.mode,
                "stride": bitmap.stride,
                "shared_memory": name,
            }
        except BaseException:
            if storage is not None:
                storage.unlink()
            raise
        finally:
            if storage is not None:
                storage.close()
            if bitmap is not None:
                bitmap.close()


def estimate_page_image_bytes(page_size: tuple[float, float], dpi: int = DEFAULT_PDF_IMAGE_DPI) -> int:
    """按现有渲染缩放估算四通道页图字节，用于限制批量驻留内存。"""
    import math

    width, height = page_size
    scale = min(dpi / 72, DEFAULT_MAX_RENDER_EDGE / max(width, height))
    return math.ceil(width * scale) * math.ceil(height * scale) * 4


def page_to_image(
    page: PdfPage,
    dpi: int = DEFAULT_PDF_IMAGE_DPI,
    max_width_or_height: int = DEFAULT_MAX_RENDER_EDGE,
) -> tuple[Image.Image, float]:
    """按既有 DPI 与长边上限渲染页面，并独立持有返回图片的像素。"""
    with pdfium_guard():
        bitmap: PdfBitmap | None = None
        try:
            bitmap, scale = _render_page_bitmap(page, dpi, max_width_or_height)
            image = bitmap.to_pil().copy()
        finally:
            if bitmap is not None:
                try:
                    bitmap.close()
                except Exception as e:
                    logger.error(f"Failed to close bitmap: {e}")
    return image, scale


__all__ = ["page_to_image"]


def page_to_owned_bitmap(page: PdfPage, dpi: int = DEFAULT_PDF_IMAGE_DPI):
    """一次复制 PDFium 位图到不可变字节，关闭句柄后供 Rust 安全裁剪和旋转。"""
    with pdfium_guard():
        bitmap = None
        try:
            bitmap, _ = _render_page_bitmap(page, dpi, DEFAULT_MAX_RENDER_EDGE)
            return bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.stride, bitmap.mode
        finally:
            if bitmap is not None:
                bitmap.close()
