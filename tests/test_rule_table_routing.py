import numpy as np

from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self import ModelType


def _model(model_type):
    model=RapidTableModel.__new__(RapidTableModel)
    model.model_type=model_type
    return model


def test_single_wired_model_is_rule_eligible_without_classifier():
    assert _model(ModelType.UNET).rule_table_class(np.zeros((10,10,3), dtype=np.uint8)) == ('wired', 1.0)


def test_single_wireless_model_skips_rules():
    assert _model(ModelType.SLANETPLUS).rule_table_class(np.zeros((10,10,3), dtype=np.uint8)) == (None, 1.0)


def test_combined_model_returns_classifier_route_once():
    model=_model(ModelType.UNET_SLANET_PLUS)
    calls=[]
    class Classifier:
        def __call__(self, images, return_scores=False):
            calls.append(images)
            return ['wireless'], [.98], 0.01
    model.table_cls=Classifier()
    assert model.rule_table_class(np.zeros((10,10,3), dtype=np.uint8)) == ('wireless', .98)
    assert len(calls) == 1
