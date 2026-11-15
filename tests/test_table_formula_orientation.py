from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from rapid_doc.backend.pipeline import analyze_utils
from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
from rapid_doc.backend.pipeline.table_formula import prepare_table_formula_crops, rotate_table_hint_boxes


@pytest.mark.parametrize('label,expected', [('90', [2, 4, 5, 7]), ('270', [3, 1, 6, 4])])
def test_rotated_formula_and_checkbox_hints_keep_content_and_input(label, expected):
    hints = [{'bbox': [1, 2, 4, 5], 'latex': r'\pm 1'},
             {'bbox': [1, 2, 4, 5], 'checkbox': '☑'}]
    original = deepcopy(hints)
    result = rotate_table_hint_boxes(hints, (8, 8, 3), label)
    assert [hint['bbox'] for hint in result] == [expected, expected]
    assert result[0]['latex'] == r'\pm 1' and result[1]['checkbox'] == '☑'
    assert hints == original


def table(poly, page=0):
    return dict(table_res={'poly': poly}, table_img=np.zeros((20, 10, 3), np.uint8),
                single_page_mfdetrec_res=[], checkbox_res=[], useful_list=[0, 0, 0, 0, 10, 20, 10, 20],
                page_idx=page, lang='ch')


@pytest.mark.parametrize('label,turns', [('90', 1), ('270', -1), ('0', 0)])
def test_only_owned_formula_is_oriented_and_recognized_once(monkeypatch, label, turns):
    crop = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
    formulas = [dict(formula_res={'bbox': box}, formula_img=crop.copy(), page_idx=page)
                for box, page in [([1, 2, 4, 8], 0), ([12, 2, 15, 8], 0), ([1, 2, 4, 8], 1)]]
    original = deepcopy([f['formula_res'] for f in formulas])
    tables = [table([0, 0, 10, 0, 10, 20, 0, 20])]
    probe = Mock(return_value=(['cached-detection'], label))
    monkeypatch.setattr(analyze_utils, '_get_table_ocr_model', Mock())
    monkeypatch.setattr(analyze_utils, '_probe_table_direction', probe)
    prepare_table_formula_crops(formulas, tables, [{}], [2], Mock(), {}, {})
    np.testing.assert_array_equal(formulas[0]['formula_img'], np.rot90(crop, turns))
    for f in formulas[1:]:
        np.testing.assert_array_equal(f['formula_img'], crop)
    assert [f['formula_res'] for f in formulas] == original
    assert tables[0]['_formula_direction_probe'] == (['cached-detection'], label)
    probe.assert_called_once()
    batch = BatchAnalyze.__new__(BatchAnalyze)
    batch.model = Mock()
    batch.formula_base_batch_size = 1
    batch.model.formula_model.batch_predict.return_value = ['A', 'B', 'C']
    batch._run_formula_recognition(formulas)
    batch.model.formula_model.batch_predict.assert_called_once()
    assert [f['formula_res']['latex'] for f in formulas] == ['A', 'B', 'C']


def test_overlapping_tables_assign_formula_to_one_smallest_owner(monkeypatch):
    tables = [table([0, 0, 10, 0, 10, 20, 0, 20]),
              table([0, 0, 6, 0, 6, 10, 0, 10])]
    crop = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)
    formulas = [dict(formula_res={'bbox': [1, 2, 4, 8]}, formula_img=crop, page_idx=0)]
    probe = Mock(return_value=([], '90'))
    monkeypatch.setattr(analyze_utils, '_get_table_ocr_model', Mock())
    monkeypatch.setattr(analyze_utils, '_probe_table_direction', probe)
    prepare_table_formula_crops(formulas, tables, [{}], [2], Mock(), {}, {})
    probe.assert_called_once()
    assert '_formula_direction_probe' not in tables[0]
    assert '_formula_direction_probe' in tables[1]
    np.testing.assert_array_equal(formulas[0]['formula_img'], np.rot90(crop))


def test_tables_without_formula_ownership_do_not_probe(monkeypatch):
    probe = Mock()
    monkeypatch.setattr(analyze_utils, '_probe_table_direction', probe)
    formulas = [dict(formula_res={'bbox': [12, 2, 15, 8]}, formula_img=np.zeros((3, 2, 3)), page_idx=0)]
    prepare_table_formula_crops(formulas, [table([0, 0, 10, 0, 10, 20, 0, 20])], [{}], [2], Mock(), {}, {})
    probe.assert_not_called()


@pytest.mark.parametrize('label,expected', [('90', [2, 6, 8, 9]), ('270', [12, 1, 18, 4])])
def test_table_stage_reuses_probe_and_matches_formula_in_rotated_coordinates(monkeypatch, label, expected):
    item = table([0, 0, 10, 0, 10, 20, 0, 20])
    formula = {'bbox': [1, 2, 4, 8], 'latex': r'\pm 1'}
    item.update(ocr_enable=True, single_page_mfdetrec_res=[formula],
                _formula_direction_probe=([], label))
    page = {'blocks': [], 'ori_image_list': []}
    table_model = Mock(spec=['predict'])
    table_model.predict.return_value = '<table><tr><td>x</td></tr></table>'
    manager = Mock()
    manager.get_atom_model.return_value = table_model
    monkeypatch.setattr(analyze_utils, '_get_table_ocr_model', Mock())
    probe = Mock(side_effect=AssertionError('direction must not be probed twice'))
    monkeypatch.setattr(analyze_utils, '_probe_table_direction', probe)
    detection = Mock(return_value=[])
    monkeypatch.setattr(analyze_utils, '_detect_table_text', detection)
    analyze_utils._process_single_table(
        item, page, 1, manager, {'use_rule_table': False, 'table_image_enable': False}, {},
    )
    probe.assert_not_called()
    # Only the post-rotation detection remains; the unrotated detection was cached.
    detection.assert_called_once()
    masks = detection.call_args.args[2]
    assert masks == [{'bbox': expected, 'latex': r'\pm 1'}]
    assert table_model.predict.call_args.args[3] == masks
    assert formula['bbox'] == [1, 2, 4, 8]
    assert '_formula_direction_probe' not in item
