from copy import deepcopy
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from rapid_doc.model.table.docvortex_ocr import recover_ocr_table, prepare_ocr_table
from rapid_doc.model.docvortex_adapter import use_docvortex_table_rules, use_docvortex_ocr_table_rules


def grid(merged=False, blank_row=False, lines=True):
    image = np.full((184,304,3),255,np.uint8)
    if lines:
        for y in (2,62,122,182):
            cv2.line(image,(2,y),(302,y),(0,0,0),1)
        for x in (2,102,202,302):
            cv2.line(image,(x,62 if merged and x == 102 else 2),(x,182),(0,0,0),1)
    boxes,texts = [],[]
    for row in range(3):
        if blank_row and row == 1:
            continue
        for col in range(3):
            if merged and row == 0 and col == 1:
                continue
            x,y = 20+col*100,20+row*60
            boxes.append([[x,y],[x+30,y],[x+30,y+16],[x,y+16]])
            texts.append('A &amp; B &lt;C&gt;' if (row,col)==(0,0) else f'R{row}C{col}')
    return image,(boxes,texts,[.99]*len(boxes))


def test_grid_and_html_escape_once():
    image,ocr = grid()
    original = deepcopy(ocr)
    result = recover_ocr_table(image,ocr)
    assert (result.rows,result.cols,len(result.cells)) == (3,3,9)
    assert 'A &amp; B &lt;C&gt;' in result.html
    assert '&amp;amp;' not in result.html
    assert ocr == original


def test_merged_and_blank_cells_are_preserved():
    image,ocr = grid(merged=True,blank_row=True)
    result = recover_ocr_table(image,ocr)
    assert result is not None
    assert (result.rows,result.cols,len(result.cells)) == (3,3,8)
    assert any(cell.colspan == 2 for cell in result.cells)
    assert sum(not c.content for c in result.cells) == 3


def test_wireless_grid_declines_without_physical_boundaries():
    image,ocr = grid(lines=False)
    result = recover_ocr_table(image,ocr)
    assert result is None


def test_every_ocr_word_must_survive_native_text_preparation():
    image,ocr=grid()
    # Native glyph preparation filters very thin Latin boxes. The adapter must
    # decline rather than return a complete-looking grid with that text lost.
    ocr[0][0]=[[20,20],[50,20],[50,20.2],[20,20.2]]
    assert recover_ocr_table(image,ocr) is None


@pytest.mark.parametrize('new_shape',[False,True])
def test_opencv4_and_opencv5_hough_shapes(monkeypatch,new_shape):
    from rapid_doc.model.table import docvortex_ocr
    image,ocr=grid()
    lines=np.array([[2,2,302,2],[2,62,302,62],[2,122,302,122]],np.int32)
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *a,**kw:lines if new_shape else lines[:,None,:])
    prepared=docvortex_ocr.prepare_ocr_table(image,ocr)
    assert prepared is not None and prepared[2]==0


def test_missing_outer_grid_boundary_declines():
    image,ocr=grid()
    image[:,300:]=255
    assert recover_ocr_table(image,ocr) is None


def test_low_confidence_missing_and_malformed_ocr_decline():
    image,ocr = grid()
    assert recover_ocr_table(image,None) is None
    ocr[2][0] = .2
    assert recover_ocr_table(image,ocr) is None
    with pytest.raises(ValueError,match='length mismatch'):
        recover_ocr_table(image,([],['x'],[]))
    with pytest.raises(ValueError,match='crop-local'):
        recover_ocr_table(image,([[[500,10],[530,30]]],['x'],[1]))


def test_ocr_words_stay_atomic_and_cross_cell_words_decline():
    image,ocr = grid()
    ocr[0][0] = [[20,20],[180,20],[180,36],[20,36]]
    prepared = prepare_ocr_table(image,ocr)
    assert len(prepared[0].chars) == len(ocr[0])
    assert prepared[0].chars[0]['char'] == 'A & B <C>'
    assert recover_ocr_table(image,ocr) is None


def test_small_skew_transforms_words_with_image():
    image,ocr = grid()
    matrix = cv2.getRotationMatrix2D((152,92),1.5,1)
    image = cv2.warpAffine(image,matrix,(304,184),borderValue=(255,255,255))
    # The synthetic outer border may be clipped by this rotation; decline is
    # valid, but the recovered word geometry must share the rectified frame.
    for idx,box in enumerate(ocr[0]):
        ocr[0][idx] = (np.c_[box,np.ones(4)]@matrix.T).tolist()
    prepared = prepare_ocr_table(image,ocr)
    assert abs(prepared[2]) > .5
    assert len(prepared[0].chars) == len(ocr[0])
    assert prepared[0].page_size != (304,184)


@pytest.mark.parametrize('config,native,raster',[
    ({'rule_table_engine':'docvortex_ocr'},True,True),
    ({'rule_table_engine':'docvortex_ocr','force_ocr':True},False,True),
    ({'rule_table_engine':'docvortex_ocr','use_rule_table':False},False,False),
    ({},True,False),
])
def test_route_flags(config,native,raster):
    assert bool(use_docvortex_table_rules(config)) is native
    assert use_docvortex_ocr_table_rules(config) is raster


def pipeline_fixture(monkeypatch):
    from rapid_doc.backend.pipeline import analyze_utils as module
    image,ocr = grid()
    table = {'lang':'ch','useful_list':[0,0,0,0,304,184,304,184],'ocr_enable':True,
             'single_page_mfdetrec_res':[],'checkbox_res':[],'table_img':image,
             'table_res':{'poly':[0,0,304,0,304,184,0,184]}}
    model = Mock()
    model.predict.return_value = '<table><tr><td>model fallback</td></tr></table>'
    manager = Mock()
    manager.get_atom_model.side_effect = lambda **kw: model if kw['atom_model_name']=='table' else Mock()
    monkeypatch.setattr(module,'_detect_table_text',lambda *a,**kw:[ocr[0][0]])
    monkeypatch.setattr(module,'txt_most_angle_extract_table',lambda *a,**kw:('0',[0]))
    monkeypatch.setattr(module,'_run_table_ocr',lambda *a,**kw:ocr)
    monkeypatch.setattr(module,'_extract_table_text_from_pdf',lambda *a,**kw:[])
    config = {'rule_table_engine':'docvortex_ocr','table_image_enable':False,'force_ocr':True}
    return module,table,manager,model,config


def test_pipeline_success_avoids_loading_model(monkeypatch):
    module,table,manager,model,config = pipeline_fixture(monkeypatch)
    module._process_single_table(table,{},1,manager,config,{})
    assert table['table_res']['table_parse_method'] == 'docvortex_ocr'
    assert not any(c.kwargs['atom_model_name']=='table' for c in manager.get_atom_model.call_args_list)
    model.predict.assert_not_called()


def test_diagnostic_failure_does_not_discard_success(monkeypatch,tmp_path):
    from rapid_doc.model.table import docvortex_ocr
    module,table,manager,model,config = pipeline_fixture(monkeypatch)
    config['rule_table_debug_dir'] = str(tmp_path)
    monkeypatch.setattr(docvortex_ocr,'save_ocr_diagnostics',Mock(side_effect=PermissionError('locked')))
    module._process_single_table(table,{},1,manager,config,{})
    assert table['table_res']['table_parse_method'] == 'docvortex_ocr'
    model.predict.assert_not_called()


@pytest.mark.parametrize('outcome',['none','exception','threshold','disabled','formula',
                                  'formula_output_disabled','image_output_disabled','low_confidence','wireless'])
def test_pipeline_fallback(monkeypatch,outcome):
    from rapid_doc.model.table import docvortex_ocr
    module,table,manager,model,config = pipeline_fixture(monkeypatch)
    if outcome == 'exception':
        monkeypatch.setattr(docvortex_ocr,'recover_ocr_table',Mock(side_effect=ValueError('bad geometry')))
    elif outcome == 'none':
        monkeypatch.setattr(docvortex_ocr,'recover_ocr_table',lambda *a,**kw:None)
    elif outcome == 'threshold':
        config['rule_table_score_threshold'] = 1.1
    elif outcome == 'disabled':
        config['use_rule_table'] = False
    elif outcome == 'formula':
        monkeypatch.setattr(module,'get_adjusted_mfdetrec_res',lambda *a,**kw:[{'bbox':[1,1,10,10],'text':'x'}])
    elif outcome == 'formula_output_disabled':
        config['table_formula_enable']=False
        table['single_page_mfdetrec_res']=[{'bbox':[20,20,50,36],'latex':'x'}]
        monkeypatch.setattr(docvortex_ocr,'recover_ocr_table',Mock(side_effect=AssertionError('Known formula must bypass OCR rules')))
    elif outcome == 'image_output_disabled':
        table['table_res']['layout_image_list']=[{'bbox':[20,20,50,36]}]
        monkeypatch.setattr(docvortex_ocr,'recover_ocr_table',Mock(side_effect=AssertionError('Known image must bypass OCR rules')))
    elif outcome == 'low_confidence':
        image,ocr=grid()
        ocr[2][0]=.2
        monkeypatch.setattr(module,'_run_table_ocr',lambda *a,**kw:ocr)
    elif outcome == 'wireless':
        table['table_img']=np.full_like(table['table_img'],255)
    module._process_single_table(table,{},1,manager,config,{})
    model.predict.assert_called_once()
    assert 'model fallback' in table['table_res']['html']
    assert table['table_res'].get('table_parse_method') != 'docvortex_ocr'
    if outcome in ('formula_output_disabled','image_output_disabled'):
        docvortex_ocr.recover_ocr_table.assert_not_called()


def test_raster_mode_delays_model_initialization_even_with_force_ocr(monkeypatch):
    from rapid_doc.backend.pipeline import model_init
    manager=Mock()
    monkeypatch.setattr(model_init,'AtomModelSingleton',lambda:manager)
    model_init.MineruPipelineModel(layout_config={},ocr_config={},formula_config={'enable':False},
                                  table_config={'rule_table_engine':'docvortex_ocr','force_ocr':True})
    assert not any(call.kwargs.get('atom_model_name')==model_init.AtomicModel.Table
                   for call in manager.get_atom_model.call_args_list)


def test_native_rule_still_precedes_ocr_in_raster_mode(monkeypatch):
    from rapid_doc.backend.pipeline import analyze_utils as module
    table={'lang':'ch','useful_list':[0,0,0,0,304,184,304,184],'ocr_enable':False,
           'single_page_mfdetrec_res':[],'checkbox_res':[],
           'table_res':{'poly':[0,0,304,0,304,184,0,184]}}
    manager=Mock()
    manager.get_atom_model.side_effect=AssertionError('Native success must not initialize OCR/table models')
    native=Mock(html='<table><tr><td>native first</td></tr></table>',confidence=.99)
    monkeypatch.setattr(module,'recover_docvortex_table',lambda *a:native)
    module._process_single_table(table,{'_source_pdf_bytes':b'pdf'},1,manager,
                                 {'rule_table_engine':'docvortex_ocr'}, {})
    assert table['table_res']['table_parse_method']=='docvortex_native'
    manager.get_atom_model.assert_not_called()
