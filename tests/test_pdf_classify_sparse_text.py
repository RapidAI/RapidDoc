from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.utils import pdf_classify as classifier


def pdf(pages):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 500))
    for draw in pages:
        draw(canvas)
        canvas.showPage()
    canvas.save()
    return stream.getvalue()


def blank(canvas):
    pass


def sparse(canvas):
    canvas.drawString(150, 400, 'Short title')


def scan(canvas):
    image = Image.new('RGB', (200, 250), '#b0b0b0')
    canvas.drawImage(ImageReader(image), 0, 0, 400, 500)


def dense(canvas):
    for row in range(10):
        canvas.drawString(20, 470-row*25, 'Readable native text on an ordinary page.')


@pytest.mark.parametrize('pages', [[sparse], [sparse, blank, blank], [blank], [blank, blank]])
def test_sparse_native_text_and_blank_pages_stay_text(pages):
    assert classifier.classify(pdf(pages)) == 'txt'


def test_small_logo_does_not_turn_a_sparse_text_page_into_a_scan():
    def draw(canvas):
        sparse(canvas)
        image = Image.new('RGB', (20, 20), '#b0b0b0')
        # Repeated overlapping logos must not inflate image coverage.
        for _ in range(8):
            canvas.drawImage(ImageReader(image), 10, 10, 140, 140)
    assert classifier.classify(pdf([draw, blank])) == 'txt'


@pytest.mark.parametrize('pages', [[scan], [scan, blank], [dense, scan, blank]])
def test_scans_are_not_hidden_by_blank_or_text_pages(pages):
    assert classifier.classify(pdf(pages)) == 'ocr'


def test_scan_with_sparse_native_page_number_still_needs_ocr():
    def draw(canvas):
        scan(canvas)
        canvas.drawString(190, 10, '1')
    assert classifier.classify(pdf([draw])) == 'ocr'


def test_nonempty_vector_page_without_text_keeps_ocr_fallback():
    assert classifier.classify(pdf([lambda c: c.rect(20, 20, 360, 460, fill=1)])) == 'ocr'


def test_sparse_bad_unicode_is_not_exempted_by_low_character_count(monkeypatch):
    collect = classifier._collect_pdfium_text_samples
    def bad_text(*args):
        samples = collect(*args)
        samples[0].update(cleaned_text='\ufffd'*10, char_count=10, replacement_char_count=10)
        return samples
    monkeypatch.setattr(classifier, '_collect_pdfium_text_samples', bad_text)
    assert classifier.classify(pdf([sparse, blank, blank])) == 'ocr'


def test_reported_text_table_with_sparse_trailing_pages():
    path = Path('C:/Users/huazhen/Downloads/test3.pdf')
    if not path.exists():
        pytest.skip('Optional local reproduction PDF is unavailable')
    assert classifier.classify(path.read_bytes()) == 'txt'


@pytest.mark.parametrize('method,expected', [('auto', False), ('txt', False), ('ocr', True)])
def test_pipeline_preserves_explicit_parse_method_overrides(monkeypatch, method, expected):
    from rapid_doc.backend.pipeline import pipeline_analyze

    seen = []
    def batch(pages, *args):
        seen.extend(item[2] for item in pages)
        return [[] for _ in pages]
    monkeypatch.setattr(pipeline_analyze, 'batch_image_analyze', batch)
    if method != 'auto':
        def fail(*args):
            pytest.fail('Explicit parsing modes must bypass automatic classification')
        monkeypatch.setattr(pipeline_analyze, 'classify', fail)
    result = pipeline_analyze.doc_analyze([pdf([sparse, blank, blank])], parse_method=method,
                                         formula_enable=False, table_enable=False)
    assert result[4] == [expected]
    assert seen == [expected] * 3
