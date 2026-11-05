"""Installed native gate and score policy agree with independent Python."""
import json
import pytest
from rapid_doc_rust import legacy_pipeline_json


def op(name, request):
    return json.loads(legacy_pipeline_json(name,json.dumps(request,allow_nan=False)))


def test_square_digit_boxes_no_longer_block_portrait_classification():
    request=dict(page={'blocks':[]},table={'poly':[0,0,100,0,100,200,0,200]},
                 width=100,height=200,boxes=[[[1,1],[11,1],[11,11],[1,11]]]*10,scale=1)
    assert op('plan_table_direction',request)['needs_classifier'] is True
    request['height']=120
    assert op('plan_table_direction',request)['needs_classifier'] is False
    request['height']=200
    request['page']['blocks']=[{'lines':[dict(rotation=0,spans=[dict(text='native',bbox=[10,10,30,30])])]}]
    assert op('plan_table_direction',request)['needs_classifier'] is False


@pytest.mark.parametrize('angle,score,margin,expected',[(270,.904,.86,'270'),(90,.95,.9,'90'),
    (90,.879,.8,'0'),(270,.91,.1,'0'),(0,.99,.98,'0'),(180,.99,.98,'0')])
def test_native_model_score_gate(angle,score,margin,expected):
    assert op('select_table_orientation',dict(rotation_degrees=angle,score=score,margin=margin))==expected
