"""General geometry/routing regressions for the six electronic PDF reports."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline, table_rules
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem, _TableCandidate
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout, _block_text


def pdf(draw, size=(600, 800)):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=size)
    draw(canvas)
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize('label', ['A short unfamiliar label', 'Arbitrary stock code 654321'])
def test_plain_text_frame_retains_native_text(label):
    def draw(c):
        c.rect(60, 420, 240, 110)
        c.drawString(70, 500, label)
        c.drawString(70, 470, 'Ordinary panel body text follows.')
    blocks = analyze_docvortex_layout(pdf(draw))[0]
    assert not any(b['type'] in {'image', 'table'} for b in blocks)
    assert label in ''.join(_block_text(b['content']) for b in blocks)


def test_sparse_foreground_text_on_full_page_background_is_not_image_content():
    def draw(c):
        c.drawImage(ImageReader(Image.new('RGB', (30, 40), '#739ac4')), 0, 0, 600, 800)
        c.drawString(250, 150, 'Contact author: Example')
        c.drawString(250, 125, 'Email: example@example.test')
    blocks = analyze_docvortex_layout(pdf(draw))[0]
    assert any(b['type'] == 'image' for b in blocks)
    texts = ''.join(_block_text(b['content']) for b in blocks if b['type'] == 'text')
    assert 'example@example.test' in texts
    assert not any('Contact author' in _block_text(b['content']) for b in blocks if b['type'] == 'image')


def test_leader_toc_with_unknown_heading_keeps_short_first_and_last_sections():
    def draw(c):
        c.setFont('Helvetica', 10)
        c.drawCentredString(300, 745, 'Unfamiliar TOC heading')
        for index, y in enumerate([710, 695, 650, 635, 620, 605, 590, 575, 560, 515, 500]):
            c.drawString(60, y, f'{index+1}. Section {index+1} ' + '.' * 75)
            c.drawRightString(520, y, f'4-{index+1}')
    blocks = analyze_docvortex_layout(pdf(draw))[0]
    assert not any(b['type'] == 'table' for b in blocks)
    content = ''.join(_block_text(b['content']) for b in blocks if b['type'] == 'index')
    for index in range(11):
        assert f'Section {index+1} ' in content
        assert f'4-{index+1}' in content


@pytest.mark.parametrize('scale', [0.75, 1, 1.4])
def test_wide_connected_raster_label_diagram_is_kept_as_one_image(scale):
    def draw(c):
        c.scale(scale, scale)
        for col in range(4):
            x = 65 + col * 115
            for row in range(2):
                y = 490 - row * 90
                c.rect(x, y, 85, 45)
                c.drawImage(ImageReader(Image.new('RGB', (25, 8), 'black')), x + 15, y + 20, 25, 8)
            c.line(x + 40, 445, x + 40, 490)
            if col:
                c.line(x - 30, 512, x, 512)
        c.drawString(65, 350, 'An ordinary paragraph outside the diagram.')
    blocks = analyze_docvortex_layout(pdf(draw, (600*scale, 800*scale)))[0]
    images = [b for b in blocks if b['type'] == 'image']
    assert len(images) == 1
    assert images[0]['bbox'][2] - images[0]['bbox'][0] > .65
    assert 'An ordinary paragraph' in ''.join(_block_text(b['content']) for b in blocks if b['type'] == 'text')


def test_fragmented_column_strokes_prove_two_separate_grids():
    def draw(c):
        c.setFont('Helvetica', 9)
        for base in [460, 600]:
            for row in range(3):
                for x in [50, 170, 290, 410]:
                    c.line(x, base + row*30, x, base + (row+1)*30)
                for col in range(3):
                    c.drawString(55 + col*120, base + row*30 + 10, f'R{row}C{col}')
            for y in range(base, base+91, 30):
                c.line(50, y, 410, y)
    with PDFDocument(pdf(draw)) as document:
        source = pipeline._collect_document_sources(document).page_sources[0]
    assert len(table_rules._native_multicolumn_grid_bboxes(source, 9, (50, 110, 410, 340))) == 2
    assert table_rules._native_multicolumn_grid_bboxes(source, 9, (55, 110, 160, 200)) == []


def test_weak_candidate_cannot_expand_confirmed_grid_over_prose():
    first = _TableCandidate(bbox=(40, 50, 540, 200), local_bbox=(40, 50, 540, 200), angle=0, score=111, line_indices={1, 2}, core_bbox=(40, 50, 540, 200))
    weak = _TableCandidate(bbox=(40, 100, 540, 700), local_bbox=(40, 100, 540, 700), angle=0, score=18, line_indices={2, 3, 4}, core_bbox=(40, 100, 540, 700))
    second = _TableCandidate(bbox=(40, 350, 540, 440), local_bbox=(40, 350, 540, 440), angle=0, score=110, line_indices={4}, core_bbox=(40, 350, 540, 440))
    merged = table_rules._merge_table_candidates([first, weak, second])
    assert [c.bbox for c in merged] == [(40, 50, 540, 200), (40, 350, 540, 440)]
    assert merged[0].line_indices == {1, 2}


def test_table_preamble_keeps_first_copy_and_changed_values():
    pages = []
    for value in ['A', 'A', 'B']:
        members = [
            _LineItem('Account report', (190, 31, 310, 39), 0, 0, effective_height=8, semantic_type='header'),
            _LineItem(f'Arbitrary metadata value {value}', (10, 41, 280, 49), 0, 1, effective_height=8, semantic_type='header'),
            _LineItem('Generated on some date', (400, 41, 590, 49), 0, 2, effective_height=8, semantic_type='header'),
        ]
        pages.append(SimpleNamespace(page_size=(600, 800), table_bboxes=[(10, 51, 590, 750)], remaining_lines=members))
    pipeline._restore_table_preamble_metadata(pages)
    assert all(line.semantic_type is None for line in pages[0].remaining_lines)
    assert all(line.semantic_type == 'header' for line in pages[1].remaining_lines)
    assert all(line.semantic_type is None for line in pages[2].remaining_lines)


@pytest.mark.parametrize('scanned', [False, True])
def test_auto_ocr_still_supplies_rule_layout_regions(monkeypatch, scanned):
    from rapid_doc.backend.pipeline import pipeline_analyze
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel

    def draw(c):
        if scanned:
            c.drawImage(ImageReader(Image.new('RGB', (60, 80), '#739ac4')), 0, 0, 600, 800)
        else:
            c.drawString(60, 680, 'Readable native text despite the OCR classifier')
    payload = pdf(draw)
    monkeypatch.setattr(pipeline_analyze, 'classify', lambda _: 'ocr')
    monkeypatch.setattr('rapid_doc.model.layout.rapid_layout.get_device', lambda: 'cpu')
    layout_model = RapidLayoutModel({'model_type': 'docvortex_rule', 'docvortex_mode': 'rule'})
    captured = []

    def batch(pages, *args):
        captured.extend(pages)
        return layout_model.batch_predict([np.asarray(p[0]) for p in pages], 1, [p[4] for p in pages])
    monkeypatch.setattr(pipeline_analyze, 'batch_image_analyze', batch)
    output, images, pages, _, ocr = pipeline_analyze.doc_analyze(
        [payload], layout_config={'model_type': 'docvortex_rule', 'docvortex_mode': 'rule'})
    assert ocr == [True]
    assert captured[0][2] is True
    assert output[0][0]['layout_dets']
    if scanned:
        assert output[0][0]['layout_dets'][0]['original_label'] == 'text'
    for image in images[0]:
        image['img_pil'].close()
    for page in pages[0]:
        for image in page.get('ori_image_list', []):
            image['pil_image'].close()


def test_ocr_layout_does_not_bypass_unreliable_font_recognition(monkeypatch):
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel
    monkeypatch.setattr('rapid_doc.model.layout.rapid_layout.get_device', lambda: 'cpu')
    model = RapidLayoutModel({'model_type': 'docvortex_rule', 'docvortex_mode': 'rule'})
    result = model.batch_predict([np.zeros((100,100,3), dtype=np.uint8)], 1, [{
        '_docvortex_ocr': True,
        '_docvortex_layout': [{'type':'table', 'bbox':[.1,.1,.9,.9], 'content':'<table><tr><td>garbled</td></tr></table>'}],
    }])[0]
    assert result[0]['original_label'] == 'table'
    assert 'html' not in result[0]


def test_unique_background_foreground_marginal_is_kept_but_repeated_header_is_not():
    pages = []
    for index in range(2):
        lines = [
            _LineItem('Repeated publication header', (40, 25, 350, 35), 0, 0, semantic_type='header'),
            _LineItem(f'Unique contact address {index}', (240, 765, 560, 775), 0, 1, semantic_type='footer'),
        ]
        pages.append(SimpleNamespace(page_size=(600, 800), remaining_lines=lines,
                                     fixed_blocks=[{'type':'image','bbox':(0,0,600,800)}]))
    pipeline._restore_background_body_marginals(pages)
    assert all(page.remaining_lines[0].semantic_type == 'header' for page in pages)
    assert all(page.remaining_lines[1].semantic_type is None for page in pages)


@pytest.mark.parametrize('owned_foreground', [True, False])
def test_image_overlap_preserves_owned_foreground_through_final_markdown(tmp_path, owned_foreground):
    from rapid_doc.model.docvortex_adapter import convert_docvortex_layout
    from rapid_doc.backend.pipeline.model_json_to_middle_json import page_model_info_to_page_info
    from rapid_doc.backend.pipeline.para_split import para_split
    from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    from rapid_doc.data.data_reader_writer import FileBasedDataWriter
    from rapid_doc.utils.enum_class import CategoryId, MakeMode

    layout = convert_docvortex_layout([
        {'type':'image', 'bbox':[0,0,1,1], 'content':''},
        {'type':'text', 'bbox':[.4,.75,.9,.8], 'content':'Email: example@example.test'},
    ], 600, 800)
    if not owned_foreground:
        layout[1].pop('native_text_region')
    layout.append({'category_id':CategoryId.OcrText, 'poly':layout[1]['poly'],
                   'text':'Email: example@example.test', 'score':1, 'original_order':1, 'original_label':'text'})
    image = Image.new('RGB', (600, 800), '#739ac4')
    page = page_model_info_to_page_info(
        {'layout_dets':layout}, {'img_pil':image,'scale':1}, {'size':(600,800),'ori_image_list':[]},
        FileBasedDataWriter(str(tmp_path/'images')), 0, ocr_enable=True, formula_enabled=False)
    para_split([page])
    markdown = union_make([page], MakeMode.MM_MD, 'images')
    image.close()
    assert ('example@example.test' in markdown) is owned_foreground
    assert '![' in markdown
