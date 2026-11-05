from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rapid_doc.model.table.fallback import quality, plain_html, TextFallback, markdown_prose
from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self import ModelType
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown, make_blocks_to_content_list
from rapid_doc.utils.enum_class import MakeMode


def rows():
    return [[[[1, 1], [10, 1], [10, 10], [1, 10]], [[12, 1], [22, 1], [22, 10], [12, 10]]], ['评卷人', '得分'], [.99, .99]]


def model(main, backup=None, text=False):
    # No model downloads; exercise the actual preparation/selection path.
    m = RapidTableModel.__new__(RapidTableModel)
    m.model_type = ModelType.TABLE_TRANSFORMER
    m.fallback_model_type = 'unet_slanet_plus' if backup is not None else None
    m.fallback_to_text = text
    m.fallback_text_loss_threshold = .2
    m._fallback_model = Mock() if backup is not None else None
    if backup is not None:
        m._fallback_model._structure.return_value = backup
    m._structure = Mock(return_value=main)
    m.ocr_engine = Mock()
    return m


def test_retention_entities_occurrences_and_threshold():
    r = [[], ['A &amp; B', '甲甲乙', 'noise'], [1, .9, .1]]
    q = quality('<table><tr><td>A &amp; B甲乙</td></tr></table>', r)
    assert (q.total, q.retained, q.valid) == (6, 5, True)
    assert not q.needs_fallback(.2)
    assert q.needs_fallback(.1)
    q = quality('<table><tr><td>甲乙</td></tr></table>', [[], ['甲甲乙'], [1]])
    assert q.needs_fallback(.2)
    assert not quality('<table><tr><td>ABCD</td></tr></table>', [[], ['ABCDE'], [1]]).needs_fallback(.2)
    assert quality('<table></table>', rows()).needs_fallback(1)


@pytest.mark.parametrize('html', [
    '<table></table>', '<table><tr><td>A</tr></table>',
    '<table><tr><td rowspan="3">A</td></tr></table>',
    '<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>',
])
def test_invalid_structure(html):
    assert not quality(html, rows()).valid


def test_span_and_placeholders():
    html = '<table><tr><td>A &amp; B</td><td rowspan="2">甲甲乙</td></tr><tr><td></td></tr></table>'
    q = quality(html, [[], ['A &amp; B', '甲甲乙'], [1, 1]])
    assert q.valid and q.total == q.retained
    r = [[], ['AB', '$AAAA$', 'image-id'], [1, 1, 1]]
    q = quality('<table><tr><td>$AAAA$ image-id B</td></tr></table>', r, ['$AAAA$', 'image-id'])
    assert (q.total, q.retained) == (2, 1)
    # A missing formula must not remove matching letters from ordinary text.
    assert quality('<table><tr><td>AB</td></tr></table>', r, ['$AAAA$', 'image-id']).retained == 2


def test_default_off_and_healthy_main_never_run_backup():
    html = '<table><tr><td>评卷人</td><td>得分</td></tr></table>'
    for m in [model(html), model(html, '<table></table>')]:
        assert m.predict(np.zeros((24, 24, 3), np.uint8), rows()) == html
        if m._fallback_model is not None:
            m._fallback_model._structure.assert_not_called()
        m.ocr_engine.ocr.assert_not_called()


def test_backup_restores_small_table_without_second_ocr_or_mutating_input():
    html = '<table><tr><td>评卷人</td><td>得分</td></tr><tr><td></td><td></td></tr></table>'
    m = model('<table></table>', html)
    r = rows(); original = deepcopy(r)
    assert m.predict(np.zeros((24, 24, 3), np.uint8), r) == html
    assert r == original
    m._fallback_model._structure.assert_called_once()
    m.ocr_engine.ocr.assert_not_called()


def test_switches_are_independent_and_reject_bad_backup():
    main = '<table><tr><td>得分</td></tr></table>'
    # More text with broken geometry is not a valid replacement.
    bad = '<table><tr><td>评卷人得分</td><td></td></tr><tr><td></td></tr></table>'
    m = model(main, bad)
    assert m.predict(np.zeros((24, 24, 3), np.uint8), rows()) == main
    for m in [model(main, bad, text=True), model(main, text=True)]:
        result = m.predict(np.zeros((24, 24, 3), np.uint8), rows())
        assert isinstance(result, TextFallback)
        assert '评卷人' in result and '得分' in result and '<table' not in result


def test_backup_is_created_only_on_demand_and_reused():
    good = '<table><tr><td>评卷人</td><td>得分</td></tr></table>'
    m = model(good, good)
    m._fallback_model = None
    m._fallback_config = dict(model_type='table_transformer', fallback_model_type='unet_slanet_plus')
    backup = Mock(); backup._structure.return_value = good
    with patch('rapid_doc.model.table.rapid_table.RapidTableModel', return_value=backup) as create:
        m.predict(np.zeros((24,24,3), np.uint8), rows())
        create.assert_not_called()
        m._structure.return_value = '<table></table>'
        for _ in range(2):
            assert m.predict(np.zeros((24,24,3), np.uint8), rows()) == good
        create.assert_called_once()
        assert 'fallback_model_type' not in create.call_args.args[1]


def test_recovered_text_exports_to_markdown_and_content_list():
    html = str(plain_html(rows()))
    span = dict(type='table', html=html, text_fallback=True)
    block = dict(type='table', bbox=[0, 0, 100, 100], blocks=[dict(type='table_body', lines=[dict(spans=[span])])])
    for mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        md = '\n'.join(make_blocks_to_markdown([block], mode))
        assert '评卷人' in md and '得分' in md and '<table' not in md
    out = make_blocks_to_content_list(block, 'images', 0, [100, 100])
    assert out['type'] == 'text' and '评卷人' in out['text']


def test_pipeline_carries_text_recovery_marker(monkeypatch):
    from rapid_doc.backend.pipeline import analyze_utils as module
    r = rows()
    table = dict(lang='ch', useful_list=[0,0,0,0,24,24,24,24], ocr_enable=True,
                 single_page_mfdetrec_res=[], checkbox_res=[], table_img=np.zeros((24,24,3),np.uint8),
                 table_res={'poly':[0,0,24,0,24,24,0,24]})
    m = Mock(); m.predict.return_value = plain_html(r)
    manager = Mock(); manager.get_atom_model.side_effect = lambda **kw:m if kw['atom_model_name']=='table' else Mock()
    monkeypatch.setattr(module,'_detect_table_text',lambda *a,**kw:r[0])
    monkeypatch.setattr(module,'txt_most_angle_extract_table',lambda *a,**kw:('0',[0,1]))
    monkeypatch.setattr(module,'_run_table_ocr',lambda *a,**kw:r)
    module._process_single_table(table,{},1,manager,{'force_ocr':True,'use_rule_table':False,'table_image_enable':False},{})
    assert table['table_res']['text_fallback'] is True
    assert '评卷人' in table['table_res']['html']


def test_prose_groups_jittered_boxes_into_rows_before_sorting_left_to_right():
    def q(x, y, height=12):
        return [[x,y],[x+20,y],[x+20,y+height],[x,y+height]]
    # Numeric tops can be above the row label; input OCR order is arbitrary.
    r = [[q(100,9),q(0,11),q(70,12),q(0,32),q(100,30),q(70,31)],
         ['35','Freecell (80)','71','GED (20)','20','10'],[1]*6]
    assert str(plain_html(r)) == '<p>Freecell (80)  71  35</p>\n<p>GED (20)  10  20</p>\n'


def test_prose_overlaps_do_not_chain_across_neighboring_rows():
    boxes = [[[x,y],[x+10,y],[x+10,y+10],[x,y+10]] for x,y in [(0,0),(20,4),(40,8)]]
    assert str(plain_html([boxes,['A','B','C'],[1]*3])) == '<p>A  B</p>\n<p>C</p>\n'


@pytest.mark.parametrize('markup,expected', [
    ('<p>A &amp; B  12</p>\n<p>C  34</p>\n', '```text\nA & B  12\n\nC  34\n```'),
    ('<p>A\u2003B</p>', '```text\nA  B\n```'),
    ('<p>```</p>', '````text\n```\n````'),
    ('<p>&lt;img&gt;</p>', '```text\n<img>\n```'),
    ('<p>$x$</p>', '$x$'),
    ('<p><img src="images/a.png" /></p>', '<img src="images/a.png" />'),
    ('', ''),
])
def test_fenced_text_preserves_spacing_and_rich_content(markup, expected):
    assert markdown_prose(markup) == expected
