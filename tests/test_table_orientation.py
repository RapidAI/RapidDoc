from unittest.mock import Mock
import numpy as np
import pytest

from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
from rapid_doc.model.orientation.rapid_orientation import RapidOrientation
from rapid_doc.backend.pipeline.analyze_utils import _recover_rotated_table_boxes, _verify_table_direction


@pytest.mark.parametrize('label,score,margin,expected', [
    ('270', .904, .86, '270'), ('90', .95, .90, '90'),
    ('90', .879, .80, '0'), ('270', .91, .10, '0'),
    ('0', .99, .98, '0'), ('180', .99, .98, '0'),
])
def test_portrait_tables_use_model_even_with_square_digit_boxes(label, score, margin, expected):
    model=RapidOrientationModel.__new__(RapidOrientationModel)
    model.orientation_engine=Mock()
    model.orientation_engine.predict_scores.return_value=(label,score,margin,.01)
    boxes=[[[0,0],[10,0],[10,10],[0,10]]]*10
    assert model.predict(np.zeros((150,100,3),np.uint8),boxes,table_mode=True)==expected
    model.orientation_engine.predict_scores.assert_called_once()


def test_landscape_and_whole_page_behavior_are_preserved():
    model=RapidOrientationModel.__new__(RapidOrientationModel)
    model.orientation_engine=Mock(return_value=('90',.01))
    assert model.predict(np.zeros((100,150,3),np.uint8),table_mode=True)=='0'
    model.orientation_engine.predict_scores.assert_not_called()
    assert model.predict(np.zeros((150,100,3),np.uint8))=='90'
    model.orientation_engine.predict_scores.assert_not_called()


def test_scores_and_legacy_return_share_one_inference():
    model=RapidOrientation.__new__(RapidOrientation)
    model.preprocess_ops=[]
    model.labels=['0','90','180','270']
    model.session=Mock(return_value=[np.asarray([[.03,.04,.03,.9]],np.float32)])
    image=np.zeros((1,1,3),np.uint8)
    label,score,margin,_=model.predict_scores(image)
    assert label=='270' and score==pytest.approx(.9) and margin==pytest.approx(.86)
    model.session.assert_called_once()
    model.session.reset_mock()
    label,elapsed=model(image)
    assert label=='270' and elapsed>=0
    model.session.assert_called_once()


@pytest.mark.parametrize('score,margin,expected', [
    (.88, .81, '270'), (.88, .30, '0'), (.69, .60, '0'), (.92, .81, '90'),
])
def test_only_uncertain_sideways_predictions_use_ocr(score, margin, expected):
    model = RapidOrientationModel.__new__(RapidOrientationModel)
    model.orientation_engine = Mock()
    model.orientation_engine.predict_scores.return_value = ('90', score, margin, .01)
    verify = Mock(return_value='270')
    assert model.predict(np.zeros((150, 100, 3), np.uint8), table_mode=True,
                         verify_direction=verify) == expected
    assert verify.call_count == int(expected == '270')
    if expected == '270':
        verify.assert_called_once_with('90')


@pytest.mark.parametrize('left,right,expected', [
    ([('??', .99)] * 2, [('Country', .99), ('Year', .98)], '270'),
    ([('12', .99)] * 2, [('21', .99)] * 2, '0'),
    ([('Name', .99), ('??', .99)], [('??', .99), ('Year', .98)], '0'),
])
def test_table_direction_uses_matching_text_crops(left, right, expected):
    model = Mock()
    model.ocr.return_value = [left + right]
    boxes = [[[x, 0], [x + 10, 0], [x + 10, 30], [x, 30]] for x in (0, 15)]
    image = np.zeros((40, 40, 3), np.uint8)
    assert _verify_table_direction(model, image, boxes, '90') == expected
    crops = model.ocr.call_args.args[0]
    assert len(crops) == 4
    assert all(crop.shape[:2] == (10, 30) for crop in crops)


@pytest.mark.parametrize('model_label,expected', [('270', '270'), ('90', '0')])
def test_numeric_evidence_can_only_confirm_the_model(model_label, expected):
    model = Mock()
    numbers = ['2017', '2014', '2013', '2010']
    model.ocr.return_value = [[(n, .79) for n in numbers] + [(n, .99) for n in numbers]]
    boxes = [[[x, 0], [x + 10, 0], [x + 10, 30], [x, 30]] for x in (0, 15, 30, 45)]
    image = np.zeros((40, 60, 3), np.uint8)
    assert _verify_table_direction(model, image, boxes, model_label) == expected


@pytest.mark.parametrize("angle", ["90", "270"])
def test_rotated_header_splits_only_with_column_evidence(angle):
    def box(x, y, w, h):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def original(rect):
        if angle == "270":
            return [[y, 159 - x] for x, y in (rect[1:] + rect[:1])]
        return [[79 - y, x] for x, y in (rect[-1:] + rect[:-1])]

    header = [box(x, 10, 28, 12) for x in (10, 40, 70, 100)]
    peers = [box(x + 3, 33, 22, 12) for x in (10, 40, 70, 100)]
    merged = box(9, 9, 120, 14)
    after = [merged, *peers]
    assert _recover_rotated_table_boxes(
        list(map(original, header + peers)), after, (160, 80), angle
    ) == header + peers
    assert _recover_rotated_table_boxes(
        list(map(original, header)), after, (160, 80), angle
    ) == after
