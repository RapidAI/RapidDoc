from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rapid_doc.model.table.fallback import quality, plain_html, TextFallback, markdown_prose
from rapid_doc.model.table.rapid_table import RapidTableModel, _TableMarkup
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


def test_shared_evidence_triggers_backup_on_raster_rules():
    image = np.full((100, 135, 3), 255, dtype=np.uint8)
    image[:, 43:45] = 0
    image[:, 87:89] = 0
    positions = [(x, y) for y in (8, 38, 68) for x in (15, 101)]
    boxes = [[(x, y), (x + 24, y), (x + 24, y + 12), (x, y + 12)] for x, y in positions]
    texts = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot']
    rows = [boxes, texts, [.99] * len(texts)]
    main_html = '<table>' + ''.join(f'<tr><td>{a}</td><td>{b}</td></tr>' for a, b in zip(texts[::2], texts[1::2])) + '</table>'
    alt_html = '<table>' + ''.join(f'<tr><td>{a}</td><td></td><td>{b}</td></tr>' for a, b in zip(texts[::2], texts[1::2])) + '</table>'
    main = _TableMarkup(main_html, SimpleNamespace(
        cell_bboxes=[[[65 * col, y - 4, 65 * col + 65, y + 26] for y in (8, 38, 68) for col in (0, 1)]],
        logic_points=[[[i, i, col, col] for i in range(3) for col in (0, 1)]],
        token_cells=[{i: (i,) for i in range(6)}],
    ))
    alt = _TableMarkup(alt_html, SimpleNamespace(
        cell_bboxes=[[[44 * col, y - 4, 44 * col + 44, y + 26] for y in (8, 38, 68) for col in (0, 1, 2)]],
        logic_points=[[[i, i, col, col] for i in range(3) for col in (0, 1, 2)]],
        token_cells=[{i: (3 * (i // 2) + 2 * (i % 2),) for i in range(6)}],
    ))
    m = model(main, alt)
    assert m.predict(image, rows) == alt_html
    m._fallback_model._structure.assert_called_once()


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


def test_structure_can_break_text_tie_but_cannot_drop_numbers():
    def box(x, y):
        return [[x, y], [x + 10, y], [x + 10, y + 8], [x, y + 8]]

    labels = ['甲组', '乙组', '丙组', '丁组', '戊组', '己组', '2,050,100']
    boxes = [box(x, y) for x, y in [(0, 0), (24, 0), (48, 0), (0, 16),
                                   (24, 16), (48, 16), (0, 32)]]
    ocr = [boxes, labels, [1] * len(labels)]
    main = ('<table><tr><td colspan="3">甲组乙组丙组</td></tr>'
            '<tr><td colspan="3">丁组戊组己组</td></tr>'
            '<tr><td colspan="3">2,050,100</td></tr></table>')
    backup = ('<table><tr><td>甲组</td><td>乙组</td><td>丙组</td></tr>'
              '<tr><td>丁组</td><td>戊组</td><td>己组</td></tr>'
              '<tr><td colspan="3">2,050,100</td></tr></table>')
    image = np.zeros((48, 64, 3), np.uint8)
    assert model(main, backup).predict(image, ocr) == backup
    rejected = backup.replace('2,050,100', '2 050 100')
    m = model(main, rejected)
    assert m.predict(image, ocr) == main
    m._fallback_model._structure.assert_called_once()

    # A better header should not hide new full-width merges in the body.
    body = '<tr><td>资料</td><td></td><td></td></tr>' * 4
    main_with_body = main.replace('</table>', body + '</table>')
    backup_with_body = backup.replace('</table>', '<tr><td colspan="3">资料</td></tr>' * 4 + '</table>')
    assert model(main_with_body, backup_with_body).predict(image, ocr) == main_with_body

    extra_row = '<tr><td></td><td></td><td></td></tr>'
    assert model(main, backup.replace('</table>', extra_row + '</table>')).predict(image, ocr) == main


@pytest.mark.parametrize('gap, expected_splits', [(2, 2), (8, 0)])
def test_split_line_evidence_uses_geometry(gap, expected_splits):
    from rapid_doc.model.table.fallback import structure_evidence

    lines = ['第一段的开头', '第一段的续行', '第二段的开头', '第二段的续行']
    boxes = [[[0, y], [80, y], [80, y + 10], [0, y + 10]]
             for y in (0, 10 + gap, 40, 50 + gap)]
    html = '<table>' + ''.join(f'<tr><td>{text}</td></tr>' for text in lines) + '</table>'
    assert structure_evidence(html, [boxes, lines, [1] * len(lines)]).split_lines == expected_splits


def test_parallel_ocr_rows_detect_collapsed_grid():
    def box(x, y):
        return [[x, y], [x + 10, y], [x + 10, y + 8], [x, y + 8]]

    labels = ['甲组', '101', '乙组', '202', '丙组', '303']
    ocr = [[box(x, y) for y in (0, 16, 32) for x in (0, 24)], labels, [1] * len(labels)]
    main = '<table><tr><td>甲组 乙组 丙组</td><td>101 202 303</td></tr></table>'
    backup = '<table><tr><td>甲组</td><td>101</td></tr><tr><td>乙组</td><td>202</td></tr><tr><td>丙组</td><td>303</td></tr></table>'
    assert model(main, backup).predict(np.zeros((48, 64, 3), np.uint8), ocr) == backup


def test_grid_comparison_counts_only_recovered_conflicts_and_penalizes_lost_assignments():
    from rapid_doc.model.table.fallback import GridEvidence

    main = GridEvidence({(0, 1), (2, 3)}, {(4, 5), (6, 7)}, 2)
    backup = GridEvidence({(4, 5)}, {(0, 1), (8, 9)}, 3)
    assert backup.compare(main) == (1, 2)


def test_model_grid_can_recover_column_with_repeated_values():
    from types import SimpleNamespace
    from rapid_doc.model.table.rapid_table import _TableMarkup

    def markup(wide):
        cells = [(0, 65)] if wide else [(0, 20), (40, 65)]
        boxes = [[x0, y, x1, y + 10] for y in (0, 15, 30) for x0, x1 in cells]
        logic = [[row, row, col, col] for row in range(3) for col in range(len(cells))]
        row = '<tr><td>10 10</td></tr>' if wide else '<tr><td>10</td><td>10</td></tr>'
        return _TableMarkup('<table>' + row * 3 + '</table>',
                            SimpleNamespace(cell_bboxes=[boxes], logic_points=[logic]))

    points = [[[x, y], [x + 8, y], [x + 8, y + 8], [x, y + 8]]
              for y in (0, 15, 30) for x in (2, 42)]
    main, backup = markup(True), markup(False)
    result = model(main, backup).predict(np.zeros((48, 70, 3), np.uint8),
                                         [points, ['10'] * 6, [1] * 6])
    assert type(result) is str and result == backup


def test_many_empty_rows_trigger_backup_but_occupied_rowspans_do_not():
    image = np.zeros((24, 24, 3), np.uint8)
    backup = '<table><tr><td>评卷人</td><td>得分</td></tr></table>'
    empty = '<tr><td></td><td></td></tr>' * 8
    main = backup.replace('</table>', empty + '</table>')
    assert model(main, backup).predict(image, rows()) == backup

    main = ('<table><tr><td rowspan="9">评卷人</td><td>得分</td></tr>'
            + '<tr><td></td></tr>' * 8 + '</table>')
    m = model(main, backup)
    assert m.predict(image, rows()) == main
    m._fallback_model._structure.assert_not_called()


def test_ruled_blank_rows_are_kept_even_when_another_defect_triggers_backup():
    image = np.full((132, 132, 3), 255, np.uint8)
    for y in range(6, 127, 20):
        image[y, 6:127] = 0
    for x in (6, 66, 126):
        image[6:127, x] = 0
    boxes = [[x, y, x + 60, y + 20] for y in range(6, 126, 20) for x in (6, 66)]
    logic = [[r, r, c, c] for r in range(6) for c in range(2)]
    backup_html = '<table><tr><td>姓名</td><td>金额</td></tr></table>'
    primary_html = backup_html.replace('</table>', '<tr><td></td><td></td></tr>' * 5 + '</table>')
    tokens = {0: (0,), 1: (1,)}
    ocr = [[[[x, 10], [x+16, 10], [x+16, 19], [x, 19]] for x in (17, 77)], ['姓名', '金额'], [.99, .99]]

    def candidate(html, boxes, logic, assignments):
        return _TableMarkup(html, SimpleNamespace(
            cell_bboxes=[boxes], logic_points=[logic], token_cells=[assignments]))

    backup = candidate(backup_html, boxes[:2], logic[:2], tokens)
    primary = candidate(primary_html, boxes, logic, tokens)
    m = model(primary, backup)
    assert m.predict(image, ocr) == primary_html
    m._fallback_model._structure.assert_not_called()

    # 文字归属异常可以触发复查。仍须保留原图有框线的填写格。
    primary = candidate(primary_html, boxes, logic, {})
    ocr[2] = [1., 1.]
    m = model(primary, backup)
    assert m.predict(image, ocr) == primary_html
    m._fallback_model._structure.assert_called_once()


def test_separated_values_in_label_cells_trigger_backup():
    def box(x, y):
        return [[x, y], [x + 12, y], [x + 12, y + 8], [x, y + 8]]

    ocr = [[box(x, y) for y in (0, 20) for x in (0, 60, 90)],
           ['第一类', '714', '(920)', '第二类', '714', '(920)'], [1] * 6]
    main = ('<table><tr><td colspan="2">第一类714</td><td>(920)</td></tr>'
            '<tr><td colspan="2">第二类(920)</td><td>714</td></tr></table>')
    backup = ('<table><tr><td>第一类</td><td>714</td><td>(920)</td></tr>'
              '<tr><td>第二类</td><td>714</td><td>(920)</td></tr></table>')
    assert model(main, backup).predict(np.zeros((48, 110, 3), np.uint8), ocr) == backup


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


def test_isolated_extra_placeholder_is_repaired_without_replacing_table():
    header = '<tr><td>评卷人</td><td>得分</td></tr>'
    normal = '<tr><td>甲</td><td>10</td></tr>'
    extra = '<tr><td></td><td>甲</td><td>10</td></tr>'
    main = '<table>' + header + normal * 9 + extra + '</table>'
    backup = '<table>' + header + '</table>'
    m = model(main, backup)
    actual = m.predict(np.zeros((24, 24, 3), np.uint8), rows())
    assert actual == main.replace(extra, normal)
    m._fallback_model._structure.assert_not_called()

    from rapid_doc.model.table.fallback import repair_local_grid
    ambiguous = '<tr><td></td><td>甲</td><td></td></tr>'
    assert repair_local_grid(main.replace(extra, ambiguous)) is None


def test_internal_ocr_rows_restore_columns_and_extend_existing_span():
    from rapid_doc.model.table.fallback import restore_internal_rows
    from rapid_doc.model.table.table_evidence import TableEvidence

    html = '<table><tr><td rowspan="2">group</td><td>A</td><td>10</td></tr><tr><td>B</td><td>20</td></tr></table>'
    boxes = [[0, 0, 30, 100], [30, 0, 60, 20], [60, 0, 90, 20],
             [30, 80, 60, 100], [60, 80, 90, 100]]
    logic = [[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 1, 1], [1, 1, 2, 2]]
    ocr = [[[ [x, y], [x+10, y], [x+10, y+8], [x, y+8] ]
            for y in (25, 41, 57) for x in (35, 65)], ['C', '30', 'D', '40', 'E', '50'], [1.] * 6]
    source = TableEvidence(np.full((108, 98, 3), 255, np.uint8), ocr)
    restored, grid, assignments = restore_internal_rows(html, (boxes, logic), {}, source, set(range(6)), ocr)
    assert '<td rowspan="5">group</td>' in restored
    assert '<tr><td>D</td><td>40</td></tr>' in restored
    assert quality(restored, ocr).valid
    assert not source.score(grid, assignments).missing
    # 坐标不再与已知列对齐时，不拼接候选。
    ocr[0][1] = [[55, 25], [65, 25], [65, 33], [55, 33]]
    source = TableEvidence(np.full((108, 98, 3), 255, np.uint8), ocr)
    assert restore_internal_rows(html, (boxes, logic), {}, source, set(range(6)), ocr) is None


def test_aligned_row_bands_use_spans_and_respect_rules():
    from rapid_doc.model.table.fallback import join_aligned_row_bands
    from rapid_doc.model.table.table_evidence import TableEvidence

    texts = [
        ['Sample', 'A', 'B', 'C'],
        *([['Group', 'North', 'Center', 'South'], ['', 'Annex', 'Hall', 'Gate']] * 3),
    ]
    html = '<table>' + ''.join(
        '<tr>' + ''.join(('<td rowspan="2">Group</td>' if col == 0 and value == 'Group' else f'<td>{value}</td>')
                        for col, value in enumerate(row) if value) + '</tr>' for row in texts
    ) + '</table>'
    boxes, logic, assignments = [], [], {}
    ocr = [[], [], []]
    for row, (y, values) in enumerate(zip((0, 24, 35, 54, 65, 84, 95), texts)):
        for col, value in enumerate(values):
            x = col * 30
            if col == 0 and not value:
                continue
            span = int(col == 0 and row > 0)
            boxes.append([x, y, x + 30, y + 16 + 11 * span])
            logic.append([row, row + span, col, col])
            if value:
                assignments[len(ocr[1])] = [len(logic) - 1]
                ocr[0].append([[x + 5, y], [x + 17, y], [x + 17, y + 16], [x + 5, y + 16]])
                ocr[1].append(value)
                ocr[2].append(1.)

    image = np.full((115, 120, 3), 255, np.uint8)
    grid = (boxes, logic)
    source = TableEvidence(image, ocr)
    result = join_aligned_row_bands(html, grid, assignments, source, ocr)
    assert result.count('<tr>') == 4
    assert result.count('<td>North<br/>Annex</td>') == 3

    unspanned = [[r0, r0, c0, c1] for r0, _, c0, c1 in logic]
    assert join_aligned_row_bands(html, (boxes, unspanned), assignments, source, ocr) is None

    for line in ((1, 37.5, 0, 120), (1, 37.5, 30, 75)):
        ruled = TableEvidence(image, ocr, vector_lines=[line])
        assert join_aligned_row_bands(html, grid, assignments, ruled, ocr) is None


@pytest.mark.parametrize('text', [False, True])
def test_partial_backup_beats_header_only_unless_text_recovery_is_enabled(text):
    main = '<table><tr><td>得分</td></tr></table>'
    backup = '<table><tr><td>评卷人</td></tr></table>'
    # Both candidates lose >20% of the same five OCR characters.
    assert quality(main, rows()).needs_fallback(.2)
    assert quality(backup, rows()).needs_fallback(.2)
    m = model(main, backup, text=text)
    r = rows()
    original = deepcopy(r)
    result = m.predict(np.zeros((24, 24, 3), np.uint8), r)
    if text:
        assert isinstance(result, TextFallback)
        assert '评卷人' in result and '得分' in result
    else:
        assert result == backup
    assert r == original
    m._fallback_model._structure.assert_called_once()
    m.ocr_engine.ocr.assert_not_called()


@pytest.mark.parametrize('content', ['得分', '得', ''])
def test_partial_backup_with_no_retention_gain_keeps_main(content):
    main = '<table><tr><td>得分</td></tr></table>'
    backup = f'<table><tr><td>{content}</td></tr></table>'
    m = model(main, backup)
    assert m.predict(np.zeros((24, 24, 3), np.uint8), rows()) == main


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
