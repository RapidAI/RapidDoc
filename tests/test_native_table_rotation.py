from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.model.docvortex_adapter import recover_docvortex_table


def _rows(result):
    assert result is not None and result.confidence >= .9
    return [[cell.get_text() for cell in row.find_all('td')]
            for row in BeautifulSoup(result.html, 'html.parser').find_all('tr')]


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('page_rotation', [0, 90, 180, 270])
def test_region_table_recovers_text_direction_without_explicit_angle(angle, page_rotation):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(500, 500))
    # More horizontal text outside the table must not influence its direction.
    canvas.drawString(10, 470, 'Unrelated horizontal page text ' * 2)
    canvas.translate(250, 250)
    canvas.rotate(angle)
    for x in (-90, -30, 30, 90):
        canvas.line(x, -40, x, 40)
    for y in (-40, 0, 40):
        canvas.line(-90, y, 90, y)
    for r in range(2):
        for c in range(3):
            canvas.drawString(-85 + c * 60, 15 - r * 40, f'cell{r}{c}')
    canvas.save()
    bbox = (160, 210, 340, 290) if angle in (0, 180) else (210, 160, 290, 340)
    data = stream.getvalue()
    if page_rotation:
        reader = PdfReader(BytesIO(data))
        writer = PdfWriter()
        writer.add_page(reader.pages[0].rotate(page_rotation))
        rotated = BytesIO()
        writer.write(rotated)
        data = rotated.getvalue()
        for _ in range(page_rotation // 90):
            x0, y0, x1, y1 = bbox
            bbox = (500-y1, x0, 500-y0, x1)
    page = {'_source_pdf_bytes': data, '_source_page_index': 0}
    result = recover_docvortex_table(page, bbox)
    assert _rows(result) == [[f'cell{r}{c}' for c in range(3)] for r in range(2)]
    # Explicit angles and detached evidence reuse remain supported.
    page['_source_pdf_bytes'] = b'closed source'
    explicit = recover_docvortex_table(page, bbox, angle=(page_rotation-angle) % 360)
    assert explicit.html == result.html


def test_reported_rotated_pdf_uses_native_table_without_loading_models():
    from rapid_doc.backend.pipeline.analyze_utils import _process_single_table

    path = Path(__file__).parent / 'unittest/pdfs/test.pdf'
    if not path.exists():
        pytest.skip('Local reproduction PDF is unavailable')
    bbox = (358.012, 304.284, 407.427, 508.857)
    x0, y0, x1, y1 = bbox
    table = {'lang': 'en', 'useful_list': [0, 0, *bbox, x1-x0, y1-y0],
             'ocr_enable': False, 'single_page_mfdetrec_res': [], 'checkbox_res': [],
             'table_res': {'poly': [x0, y0, x1, y0, x1, y1, x0, y1]}}
    manager = Mock()
    manager.get_atom_model.side_effect = AssertionError('Native recovery must not load models')
    page = {'_source_pdf_bytes': path.read_bytes(), '_source_page_index': 0}
    _process_single_table(table, page, 1, manager, {}, {})
    assert table['table_res']['table_parse_method'] == 'docvortex_native'
    result = BeautifulSoup(table['table_res']['html'], 'html.parser')
    rows = [[cell.get_text() for cell in row.find_all('td')] for row in result.find_all('tr')]
    assert rows[0] == ['Model', 'R2', 'Testing Error']
    assert len(rows) == 4 and all(len(row) == 3 for row in rows)
    assert [row[0] for row in rows[1:]] == ['Linear Regression', 'Linear Regression', 'Gray Prediction']
    assert rows[1][1:] == ['0.98740', '1321.2']
    assert rows[2][1:] == ['0.98740', '1321.2']
    assert rows[3][1:] == ['0.00617', '687']
    manager.get_atom_model.assert_not_called()


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('font', ['Helvetica', 'Times-Roman', 'Courier', 'Helvetica-Oblique'])
def test_rotated_table_preserves_numbers_and_real_word_spaces(angle, font):
    expected = [['1321.2', '0.00617', '-1,321.20'],
                ['Linear Regression', '12 34', '1.2e-3'],
                ['12 kg', '2026-10-20', 'A  B']]
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(600, 600))
    canvas.translate(300, 300)
    canvas.rotate(angle)
    canvas.setFont(font, 10)
    for x in (-210, -70, 70, 210):
        canvas.line(x, -90, x, 90)
    for y in (-90, -30, 30, 90):
        canvas.line(-210, y, 210, y)
    for r, row in enumerate(expected):
        for c, text in enumerate(row):
            canvas.drawString(-200 + c * 140, 60-r*60, text)
    canvas.save()
    bbox = (90, 210, 510, 390) if angle in (0, 180) else (210, 90, 390, 510)
    result = recover_docvortex_table({'_source_pdf_bytes': stream.getvalue(), '_source_page_index': 0}, bbox)
    # Whitespace runs are normalized, but word/number boundaries must survive.
    expected[2][2] = 'A B'
    assert _rows(result) == expected
