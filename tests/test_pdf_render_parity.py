"""Compare the native renderer with Python PDFium on identical PDF bytes."""

from io import BytesIO
from contextlib import closing

import numpy as np
import pytest
from PIL import Image
import pypdfium2 as pdfium

from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes

sdk = pytest.importorskip("rapid_doc_rust")
_native = pytest.importorskip("rapid_doc_rust._native")
from rapid_doc_rust._compat import _read_input


@pytest.fixture(scope="module")
def pdf_bytes():
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    utils = pytest.importorskip("reportlab.lib.utils")
    # Fractional page dimensions and raster detail expose the difference between
    # PDFium's integer viewport and matrix rendering paths.
    y, x = np.indices((181, 239))
    pixels = np.stack([(x * 13) % 256, (y * 17) % 256, ((x + y) * 7) % 256], axis=-1)
    image = Image.fromarray(pixels.astype(np.uint8))
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(323.17, 251.29))
    for rotation in (0, 90, 180, 270):
        pdf.setPageRotation(rotation)
        pdf.drawImage(utils.ImageReader(image), 17.23, 36.51, width=173.37, height=131.19)
        pdf.drawString(20, 215, "PDFium rendering parity")
        pdf.acroForm.textfield(name=f"field_{rotation}", value="Form text", x=20, y=10,
                              width=150, height=18)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.mark.parametrize("page_index", range(4))
@pytest.mark.parametrize("dpi", [144.0, 200.0])
@pytest.mark.parametrize("rgb", [True, False])
def test_native_render_matches_python_pixels(pdf_bytes, page_index, dpi, rgb):
    assert_render_pixels(pdf_bytes, page_index, dpi, rgb)


def assert_render_pixels(pdf_bytes, page_index, dpi, rgb=True):
    library, data = sdk._prepare_pdf(pdf_bytes, None, 256 * 1024 * 1024)
    width, height, png = _native.render_pdf_page_bytes(
        library, data, page_index=page_index, dpi=dpi, rgb=rgb
    )
    mode = "RGB" if rgb else "RGBA"
    options = {} if rgb else {"force_bitmap_format": pdfium.raw.FPDFBitmap_BGRA}
    with pdfium.PdfDocument(data) as document:
        with closing(document[page_index]) as page:
            with closing(page.render(scale=dpi / 72, **options)) as bitmap:
                expected = np.array(bitmap.to_pil().convert(mode))
    with Image.open(BytesIO(png)) as image:
        actual = np.array(image.convert(mode))
    assert (width, height) == (expected.shape[1], expected.shape[0])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dpi", [144.0, 200.0])
def test_full_page_raster_matches_python_pixels(dpi):
    y, x = np.indices((960, 1280))
    pixels = np.stack([(x * 13) % 256, (y * 17) % 256, ((x + y) * 7) % 256], axis=-1)
    buffer = BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(
        buffer, format="PDF", resolution=200, quality=95, subsampling=0
    )
    assert_render_pixels(buffer.getvalue(), 0, dpi)


def test_image_input_uses_python_encoding():
    y, x = np.indices((81, 123))
    pixels = np.stack([(x * 13) % 256, (y * 17) % 256, ((x + y) * 7) % 256], axis=-1)
    image = BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(image, format="PNG")
    data = image.getvalue()
    native_pdf = _read_input(data, 0)[1]
    python_pdf = images_bytes_to_pdf_bytes(data)
    if isinstance(python_pdf, dict):
        python_pdf = python_pdf["pdf_bytes"]
    with pdfium.PdfDocument(native_pdf) as native, pdfium.PdfDocument(python_pdf) as python:
        with closing(native[0]) as native_page, closing(python[0]) as python_page:
            with closing(native_page.render(scale=200 / 72)) as a, closing(python_page.render(scale=200 / 72)) as b:
                np.testing.assert_array_equal(np.array(a.to_pil()), np.array(b.to_pil()))
