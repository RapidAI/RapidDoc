from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from rapid_doc.model.table.rapid_table_self import RapidTable, RapidTableInput, ModelType, EngineType
from rapid_doc.model.table.rapid_table_self.model_processor.main import ModelProcessor
from rapid_doc.model.table.rapid_table_self.utils import DownloadFile
from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer import main as tatr

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'rapid_doc/model/table/rapid_table_self/models/table_transformer_v1_1_all.onnx'


def predictions(merge=False):
    objects = [(0,[0,0,100,100]),(1,[0,0,50,100]),(1,[50,0,100,100]),
               (2,[0,0,100,50]),(2,[0,50,100,100])]
    if merge:
        objects += [(3,[0,0,100,50]),(5,[0,0,100,50])]
    logits = np.full((1,125,7),-10,np.float32)
    logits[:,:,6] = 10
    boxes = np.zeros((1,125,4),np.float32)
    for idx,(label,(x0,y0,x1,y1)) in enumerate(objects):
        logits[0,idx] = -10
        logits[0,idx,label] = 10
        boxes[0,idx] = [(x0+x1)/200,(y0+y1)/200,(x1-x0)/100,(y1-y0)/100]
    return logits,boxes


def test_decode_text_merge_and_input_immutability():
    tokens = tatr.ocr_to_tokens(([[[10,10],[30,20]],[[60,10],[80,20]],
                                 [[10,60],[30,70]],[[60,60],[80,70]]],
                                ['A & B','<C>','中文','123'],[1]*4))
    original = deepcopy(tokens)
    result = tatr.decode(*predictions(),(100,100),tokens)
    assert [c.text for c in ET.fromstring(result[0]).iter('td')] == ['A & B','<C>','中文','123']
    assert tokens == original
    np.testing.assert_array_equal(result[2],[[0,0,0,0],[0,0,1,1],[1,1,0,0],[1,1,1,1]])
    result = tatr.decode(*predictions(True),(100,100),[])
    assert '<th colspan="2"></th>' in result[0]
    assert len(result[1]) == 3


def test_empty_invalid_and_missing_rows():
    logits,boxes = predictions()
    logits[:]=-10
    logits[:,:,6]=10
    result=tatr.decode(logits,boxes,(100,100),[])
    assert result[0]=='<table></table>' and result[1].shape==(0,4)
    with pytest.raises(ValueError,match='shapes'):
        tatr.decode(logits[:,:5],boxes,(100,100),[])
    logits[0,0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        tatr.decode(logits,boxes,(100,100),[])
    logits,boxes=predictions()
    logits[0,3:5]=-10
    logits[0,3:5,6]=10
    assert tatr.decode(logits,boxes,(100,100),[])[0]=='<table></table>'


def test_preprocess_and_token_validation():
    image=np.zeros((200,400,3),np.uint8)
    image[:,:,2]=255
    pixels=tatr.preprocess(image)
    assert pixels.shape==(1,3,500,1000)
    np.testing.assert_allclose(pixels[0,:,0,0],[(1-.485)/.229,-.456/.224,-.406/.225],rtol=1e-6)
    assert tatr.ocr_to_tokens(None)==[]
    with pytest.raises(ValueError,match='length mismatch'):
        tatr.ocr_to_tokens(([],['x'],[]))
    with pytest.raises(ValueError,match='nonempty'):
        tatr.preprocess(np.empty((0,2,3),np.uint8))


def test_geometry_matches_independent_upstream_operations():
    from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer.geometry import Rect
    fitz = pytest.importorskip('fitz')
    rng = np.random.default_rng(42)
    pairs = [([0,0,0,0],[1,2,3,4]),([1,2,3,4],[0,0,0,0])]
    for _ in range(100):
        pairs.append((rng.uniform(-5,100,4).tolist(),rng.uniform(-5,100,4).tolist()))
    for a,b in pairs:
        for operation in ('intersect','include_rect'):
            actual = getattr(Rect(a),operation)(b)
            expected = getattr(fitz.Rect(a),operation)(b)
            assert list(actual) == list(expected)
            assert actual.get_area() == expected.get_area()


def test_model_lookup_and_engine_checks(tmp_path,monkeypatch):
    monkeypatch.setattr(ModelProcessor,'DEFAULT_MODEL_DIR',tmp_path)
    download=Mock(side_effect=lambda cfg:cfg.save_path.write_bytes(b'onnx-placeholder'))
    monkeypatch.setattr(DownloadFile,'run',download)
    assert ModelProcessor.get_model_path(ModelType.TABLE_TRANSFORMER)==str(tmp_path/MODEL.name)
    params=download.call_args.args[0]
    assert params.file_url=='https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/master/table/table_transformer_v1_1_all.onnx'
    assert params.sha256=='2c361306169b92b8fdf1ae677b0bbcc5111503729b598c388e5ef20237dc9085'
    assert (tmp_path/MODEL.name).read_bytes()==b'onnx-placeholder'
    with pytest.raises(ValueError,match='ONNXRUNTIME'):
        tatr.TableTransformerRecognition({'engine_type':EngineType.TORCH})
    session=Mock()
    session.get_output_names.return_value=['wrong']
    monkeypatch.setattr(tatr,'get_engine',lambda _:lambda cfg:session)
    with pytest.raises(ValueError,match='export_table_transformer.py'):
        tatr.TableTransformerRecognition({})


def test_batches_keep_ocr_alignment(monkeypatch):
    session=Mock(return_value=predictions())
    session.get_output_names.return_value=['logits','pred_boxes']
    monkeypatch.setattr(tatr,'get_engine',lambda _:lambda cfg:session)
    engine=RapidTable(RapidTableInput(model_type=ModelType.TABLE_TRANSFORMER,
                                     model_dir_or_path='unused',use_ocr=False))
    ocr=([[[10,10],[30,10],[30,20],[10,20]]],['hello'],[1])
    image=np.zeros((100,100,3),np.uint8)
    results=engine([image,image,image],[ocr,None,([],[],[])],batch_size=2)
    assert len(results.pred_htmls)==3 and 'hello' in results.pred_htmls[0]
    assert 'hello' not in results.pred_htmls[1]
    assert len(results.cell_bboxes)==len(results.logic_points)==3
    with pytest.raises(ValueError,match='batch length mismatch'):
        engine.table_structure([image],[])

    # Missing OCR must retain the grids, not silently discard another image's text.
    result=engine([image,image],batch_size=2)
    assert len(result.pred_htmls)==2 and all('<td>' in html for html in result.pred_htmls)
    with pytest.raises(ValueError,match='Batch size mismatch'):
        engine([image,image],[ocr])


def test_fully_spanned_rows_are_still_serialized():
    cells=[dict(row_nums=[0,1],column_nums=[0,1],bbox=[0,0,100,100],**{'cell text':'merged'}),
           dict(row_nums=[2],column_nums=[0],bbox=[0,100,50,150],**{'cell text':'last'})]
    html,boxes,logic=tatr.cells_to_result(cells,(100,150))
    assert len(ET.fromstring(html).findall('tr'))==3
    assert '<tr></tr>' in html
    assert 'rowspan="2"' in html and 'colspan="2"' in html
    np.testing.assert_array_equal(logic,[[0,1,0,1],[2,2,0,0]])


def test_low_confidence_table_does_not_override_real_table():
    logits,boxes=predictions()
    logits[0,5]=-10
    logits[0,5,0]=0.1
    logits[0,5,6]=0
    logits[0,5,1:6]=0
    boxes[0,5]=[.5,.5,2,2]
    assert tatr.decode(logits,boxes,(100,100),[])[0]==tatr.decode(*predictions(),(100,100),[])[0]
    logits[0,0]=0
    logits[0,0,0]=.1
    assert tatr.decode(logits,boxes,(100,100),[])[0]=='<table></table>'


def test_high_level_empty_ocr_and_external_input_immutability(monkeypatch):
    from rapid_doc.model.table.rapid_table import RapidTableModel
    session=Mock(return_value=predictions())
    session.get_output_names.return_value=['logits','pred_boxes']
    monkeypatch.setattr(tatr,'get_engine',lambda _:lambda cfg:session)
    ocr_engine=Mock()
    ocr_engine.ocr.return_value=([],0)
    high=RapidTableModel(ocr_engine,{'model_type':'table_transformer','model_dir_or_path':'unused'})
    image=np.zeros((100,100,3),np.uint8)
    assert '<td>'in high.predict(image)
    ocr=([[[10,10],[30,10],[30,20],[10,20]]],['A&amp;B'],[1])
    original=deepcopy(ocr)
    result=high.predict(image,ocr_result=ocr,mfd_res=[{'bbox':[55,55,85,75],'latex':'a<b'}])
    assert ocr==original
    assert 'A&amp;B'in result and 'amp;amp;'not in result
    assert 'a&lt;b'in result


@pytest.mark.parametrize('config,expected',[
    ({},False),
    ({'model_type':ModelType.TABLE_TRANSFORMER},False),
    ({'model_type':'table_transformer'},False),
    ({'model_type':'table_transformer','use_word_box':False},False),
    ({'model_type':'table_transformer','use_word_box':True},True),
    ({'model_type':ModelType.SLANETPLUS},False),
    ({'model_type':ModelType.SLANETPLUS,'use_word_box':True},True),
    ({'rule_table_engine':'docvortex_ocr','use_rule_table':True},False),
    ({'rule_table_engine':'docvortex_ocr','use_rule_table':True,'use_word_box':True},True),
    ({'model_type':'table_transformer','rule_table_engine':'docvortex_ocr','use_rule_table':True},False),
    ({'model_type':'table_transformer','rule_table_engine':'docvortex_ocr','use_rule_table':True,'use_word_box':False},False),
    ({'model_type':'table_transformer','rule_table_engine':'docvortex_ocr','use_rule_table':True,'use_word_box':True},True),
])
def test_pdf_table_route_preserves_word_box_default_and_explicit_option(monkeypatch,config,expected):
    from rapid_doc.backend.pipeline import analyze_utils as pipeline
    box=[[10,10],[30,10],[30,20],[10,20]]
    flags=[]
    table_model=SimpleNamespace(predict=Mock(return_value='<table><tr><td>x</td></tr></table>'))
    manager=Mock()
    manager.get_atom_model.side_effect=lambda **kw:table_model if kw['atom_model_name']=='table' else Mock()
    monkeypatch.setattr(pipeline,'_detect_table_text',lambda *args,**kwargs:[box])
    monkeypatch.setattr(pipeline,'txt_most_angle_extract_table',lambda *args,**kwargs:('0',[0]))
    def recognize(model,image,boxes,use_word_box):
        flags.append(use_word_box)
        return ([box],['x'],[1])
    monkeypatch.setattr(pipeline,'_run_table_ocr',recognize)
    table={'lang':'ch','useful_list':[0,0,0,0,100,100,100,100],'ocr_enable':True,
           'table_img':np.zeros((100,100,3),np.uint8),
           'single_page_mfdetrec_res':[],'checkbox_res':[],
           'table_res':{'poly':[0,0,100,0,100,100,0,100]}}
    pipeline._process_single_table(table,{},1,manager,{
        'use_rule_table':False,**config,'table_formula_enable':False,'table_image_enable':False}, {})
    assert flags==[expected]
    assert '<td>x</td>'in table['table_res']['html']


@pytest.mark.skipif(not MODEL.exists(),reason='Export Table Transformer first')
def test_real_onnx_and_high_level_escaping():
    import cv2
    from rapid_doc.model.table.rapid_table import RapidTableModel
    image=cv2.imread(str(ROOT/'tests/fixtures/table_transformer/basic.png'))
    if image is None:
        pytest.skip('Local crop fixture missing')
    engine=RapidTable(RapidTableInput(model_type=ModelType.TABLE_TRANSFORMER,use_ocr=False,
                                     engine_cfg={'intra_op_num_threads':4}))
    result=engine(image)
    assert len(result.cell_bboxes[0])>0
    x0,y0,x1,y1=result.cell_bboxes[0][0]
    ocr=([[[x0+1,y0+1],[x1-1,y0+1],[x1-1,y1-1],[x0+1,y1-1]]],['A&amp;B'],[1])
    high=RapidTableModel(Mock(),{'model_type':ModelType.TABLE_TRANSFORMER,
                                'engine_cfg':{'intra_op_num_threads':4}})
    html=high.predict(image[:,:,::-1],ocr_result=ocr)
    assert 'A&amp;B' in html and 'amp;amp;' not in html
    assert high.rule_table_class(image)==(None,1.0)
