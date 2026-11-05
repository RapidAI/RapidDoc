from unittest.mock import Mock
import numpy as np
import pytest

from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
from rapid_doc.model.orientation.rapid_orientation import RapidOrientation


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
