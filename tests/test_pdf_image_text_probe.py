import io
from pathlib import Path

import pytest
from PIL import Image
import pypdfium2 as pdfium

fitz = pytest.importorskip('fitz')
from rapid_doc.utils import pdf_image_text_probe as probe
from rapid_doc.utils.pdf_classify import classify, _get_pdfium_page_object_bounds


def make_pdf(kind='background', rotation=0, pages=1):
    image = Image.new('RGB', (300, 400), (225, 235, 245))
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    with fitz.open() as document:
        for index in range(pages):
            page = document.new_page(width=300, height=400)
            actual_kind = kind if index == 0 else 'text'
            if actual_kind == 'background':
                page.insert_image(page.rect, stream=stream.getvalue())
            for line in range(8):
                page.insert_text((20, 40 + line * 25), 'Normal native text stays readable.', fontsize=10)
            if actual_kind == 'covered':
                page.insert_image(page.rect, stream=stream.getvalue())
            if actual_kind == 'overlapping_small_images':
                for _ in range(3):
                    page.insert_image(fitz.Rect(10, 200, 200, 390), stream=stream.getvalue())
            page.set_rotation(rotation)
        return document.tobytes()


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('kind,expected', [('covered', True), ('background', False)])
def test_visible_text_and_covered_layer(rotation, kind, expected):
    with pdfium.PdfDocument(make_pdf(kind, rotation)) as document:
        page = document[0]
        try:
            original = probe._render(page, 1)
            try:
                signal = probe.image_backed_text_signal(page, _get_pdfium_page_object_bounds)
                assert signal['probed']
                assert signal['image_backed_ocr'] is expected
                after = probe._render(page, 1)
                try:
                    assert original.tobytes() == after.tobytes()
                finally:
                    after.close()
            finally:
                original.close()
        finally:
            page.close()


@pytest.mark.parametrize('kind', ['text', 'overlapping_small_images'])
def test_ordinary_pages_skip_render_probe(monkeypatch, kind):
    def fail(*args):
        pytest.fail('No rendering is needed for this page')
    monkeypatch.setattr(probe, '_difference_without_text', fail)
    with pdfium.PdfDocument(make_pdf(kind)) as document:
        page = document[0]
        try:
            assert not probe.image_backed_text_signal(page, _get_pdfium_page_object_bounds)['probed']
        finally:
            page.close()


def test_render_failure_restores_native_text(monkeypatch):
    with pdfium.PdfDocument(make_pdf('background')) as document:
        page = document[0]
        render = probe._render
        before = render(page, 1)
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated render failure')
            return render(*args)
        monkeypatch.setattr(probe, '_render', fail_second)
        try:
            with pytest.raises(RuntimeError, match='simulated'):
                probe.image_backed_text_signal(page, _get_pdfium_page_object_bounds)
            after = render(page, 1)
            try:
                assert before.tobytes() == after.tobytes()
            finally:
                after.close()
        finally:
            before.close()
            page.close()


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_cropbox_keeps_visible_background_text(rotation):
    with fitz.open(stream=make_pdf('background'), filetype='pdf') as source:
        source[0].set_cropbox(fitz.Rect(10, 15, 290, 380))
        source[0].set_rotation(rotation)
        data = source.tobytes()
    with pdfium.PdfDocument(data) as document:
        page = document[0]
        try:
            signal = probe.image_backed_text_signal(page, _get_pdfium_page_object_bounds)
            assert signal['probed']
            assert not signal['image_backed_ocr']
        finally:
            page.close()


@pytest.mark.parametrize('kind,expected', [('covered', 'ocr'), ('background', 'txt'), ('text', 'txt')])
def test_classification_routes(kind, expected):
    assert classify(make_pdf(kind)) == expected


def test_mixed_document_detects_scanned_sample():
    assert classify(make_pdf('covered', pages=3)) == 'ocr'


def test_real_pdf05_routes_to_ocr():
    path = Path(__file__).resolve().parents[1] / 'demo/pdfs/05.pdf'
    if not path.exists():
        pytest.skip('Optional real PDF fixture is not present')
    assert classify(path.read_bytes()) == 'ocr'
