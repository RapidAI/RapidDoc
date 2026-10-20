from io import BytesIO
from unittest.mock import Mock

import pytest
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.model.docvortex_adapter import recover_docvortex_table, use_docvortex_table_rules
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.vector import (
    _CanonicalTrack, _MergedRule, _collapse_outer_duplicate_tracks,
)


@pytest.mark.parametrize('config,expected', [
    (None, True), ({}, True), ({'rule_table_engine': 'docvortex'}, True),
    ({'use_rule_table': False}, False), ({'force_ocr': True}, False),
    ({'rule_table_engine': 'legacy'}, False),
    ({'rule_table_engine': 'docvortex', 'use_rule_table': False}, False),
])
def test_native_table_route_defaults_and_opt_out(config, expected):
    assert bool(use_docvortex_table_rules(config)) is expected


def table_pdf(scale):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(220*scale, 200*scale))
    canvas.scale(scale, scale)
    canvas.setLineWidth(.8)
    # Stroke centre lines and their endpoints differ by less than a line width.
    for y in (150, 120, 90, 60):
        canvas.line(19.4, y, 180.6, y)
    for x in (20, 75, 125, 180):
        canvas.line(x, 59.4, x, 150.6)
    for r in range(3):
        for c, x in enumerate((25, 80, 130)):
            canvas.drawString(x, 132-r*30, f'{r}{c}')
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize('scale', [.6, 1, 2])
@pytest.mark.parametrize('offset', [-.4, 0, .4])
def test_offset_outer_endpoints_preserve_table_cells(scale, offset):
    data = table_pdf(scale)
    bbox = tuple(v*scale for v in (19.2+offset, 49.2+offset, 180.8+offset, 140.8+offset))
    result = recover_docvortex_table({'_source_pdf_bytes': data, '_source_page_index': 0}, bbox)
    assert result and result.confidence >= .9
    assert result.html.count('<tr>') == 3
    assert result.html.count('<td') == 9
    assert all(f'>{r}{c}</td>' in result.html for r in range(3) for c in range(3))


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
def test_close_but_distinct_physical_borders_are_not_collapsed(orientation):
    tracks = [_CanonicalTrack(v, (v,)) for v in (0, 1.2, 30, 60)]
    rules = [_MergedRule(orientation, v, 0, 100) for v in (0, 1.2, 30, 60)]
    # Both close lines fall within snap tolerance of both tracks.
    kept, collapsed, conflict = _collapse_outer_duplicate_tracks(
        tracks, [], 5, 60, rules, orientation, 100, 1.5, True)
    assert [t.coordinate for t in kept] == [0, 1.2, 30, 60]
    assert collapsed == 0 and not conflict


def test_outer_band_with_text_is_not_collapsed():
    tracks = [_CanonicalTrack(v, (v,)) for v in (0, 1.2, 30, 60)]
    rules = [_MergedRule('horizontal', v, 0, 100) for v in (.6, 30, 60)]
    kept, collapsed, conflict = _collapse_outer_duplicate_tracks(
        tracks, [.6, 15, 45], 5, 60, rules, 'horizontal', 100, 1, True)
    assert [t.coordinate for t in kept] == [0, 1.2, 30, 60]
    assert collapsed == 0 and not conflict


def test_default_table_processing_uses_native_html_without_loading_model():
    from rapid_doc.backend.pipeline.analyze_utils import _process_single_table
    data = table_pdf(1)
    table = {'lang': 'ch', 'useful_list': [0,0,19.2,49.2,180.8,140.8,162,92], 'ocr_enable': False,
             'single_page_mfdetrec_res': [], 'checkbox_res': [],
             'table_res': {'poly': [19.2,49.2,180.8,49.2,180.8,140.8,19.2,140.8]}}
    manager = Mock()
    manager.get_atom_model.side_effect = AssertionError('Native success must not load the table model')
    _process_single_table(table, {'_source_pdf_bytes':data, '_source_page_index':0}, 1, manager, {}, {})
    assert table['table_res']['table_parse_method'] == 'docvortex_native'
    assert '>22</td>' in table['table_res']['html']
    manager.get_atom_model.assert_not_called()


@pytest.mark.parametrize('config,expect_source', [({},True),({'use_rule_table':False},False),
                                               ({'force_ocr':True},False),({'rule_table_engine':'legacy'},False)])
def test_page_preparation_uses_the_same_native_route(monkeypatch,config,expect_source):
    from rapid_doc.backend.pipeline import pipeline_analyze
    seen=[]
    def batch(items,*args):
        for image,scale,ocr,lang,page in items:
            seen.append('_source_pdf_bytes' in page and '_source_page_index' in page)
        return [[] for item in items]
    monkeypatch.setattr(pipeline_analyze,'batch_image_analyze',batch)
    pipeline_analyze.doc_analyze([table_pdf(1)],parse_method='txt',formula_enable=False,
                                table_config=config)
    assert seen == [expect_source]


@pytest.mark.parametrize('config,expect_model', [({},False),({'use_rule_table':False},True),
                                               ({'force_ocr':True},True),({'rule_table_engine':'legacy'},True)])
def test_initialization_only_loads_table_model_when_needed(monkeypatch,config,expect_model):
    from rapid_doc.backend.pipeline import model_init
    manager=Mock()
    monkeypatch.setattr(model_init,'AtomModelSingleton',lambda:manager)
    model_init.MineruPipelineModel(layout_config={},ocr_config={},formula_config={'enable':False},
                                  table_config=config)
    names=[call.kwargs.get('atom_model_name') for call in manager.get_atom_model.call_args_list]
    assert (model_init.AtomicModel.Table in names) is expect_model
