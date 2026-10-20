from copy import deepcopy

import pytest
from PIL import Image

from rapid_doc.backend.pipeline.model_json_to_middle_json import page_model_info_to_page_info
from rapid_doc.backend.pipeline.para_split import para_split
from rapid_doc.backend.pipeline.pipeline_magic_model import MagicModel
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.model.docvortex_adapter import convert_docvortex_layout
from rapid_doc.utils.enum_class import CategoryId, MakeMode


def _layout(owned=True, ocr_bbox=(279, 70.92, 326.88, 107.28), order=0):
    # Recorded title geometry: native text covers only 36% of the OCR box.
    native = (285.81, 82.51, 321.53, 100.2)
    layout = convert_docvortex_layout([
        {'type': 'text', 'bbox': [native[0]/595, native[1]/841, native[2]/595, native[3]/841],
         'content': '测试'},
        {'type': 'table', 'bbox': [.13, .143, .867, .87],
         'content': '<table><tr><td>Table body</td></tr></table>'},
    ], 595, 841)
    if not owned:
        layout[0].pop('native_text_region')
    x0, y0, x1, y1 = ocr_bbox
    layout.append({'category_id': CategoryId.OcrText, 'original_label': 'text',
                   'original_order': order, 'poly': [x0,y0,x1,y0,x1,y1,x0,y1],
                   'text': '测试', 'score': 1})
    return layout


def test_expanded_ocr_title_survives_final_markdown(tmp_path):
    image = Image.new('RGB', (595, 841), 'white')
    try:
        page = page_model_info_to_page_info(
            {'layout_dets': _layout()}, {'img_pil': image, 'scale': 1},
            {'size': (595, 841), 'ori_image_list': []},
            FileBasedDataWriter(str(tmp_path/'images')), 0, ocr_enable=True, formula_enabled=False)
        para_split([page])
        markdown = union_make([page], MakeMode.MM_MD, 'images')
        assert markdown.startswith('测试\n\n')
        assert 'Table body' in markdown
        assert markdown.count('测试') == 1
    finally:
        image.close()


@pytest.mark.parametrize('owned,order,bbox,changed', [
    (True, 0, (279,70.92,326.88,107.28), True),
    (False, 0, (279,70.92,326.88,107.28), False),
    (True, 99, (279,70.92,326.88,107.28), False),
    (True, 0, (286,83,300,99), False),  # Ordinary line already inside its region.
    (True, 0, (10,10,50,30), False),  # Wrong geometry cannot borrow ownership.
    (True, 0, (250,60,287,85), False),  # Tiny accidental corner overlap.
])
def test_only_expanded_owned_ocr_geometry_is_constrained(owned, order, bbox, changed):
    layout = _layout(owned, bbox, order)
    original = deepcopy(layout)
    spans = MagicModel({'layout_dets': layout}, 1).get_all_spans()
    span = next(s for s in spans if s.get('content') == '测试')
    assert (span['bbox'] != list(bbox)) is changed
    if changed:
        assert span['bbox'] == layout[0]['bbox']
    # Fixing detached span geometry must not rewrite the original OCR evidence.
    assert layout[-1]['poly'] == original[-1]['poly']
