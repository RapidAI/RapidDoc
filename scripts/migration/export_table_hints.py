"""冻结实际旧公式/图片提示及RapidTableModel结构输入；结构模型固定为捕获器，不执行。"""
import copy,json,math,random
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json
from export_table_text import norm,pixel

def main():
    target=ROOT/'build/migration/table-hints.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.utils.span_pre_proc as sp
    from rapid_doc.utils.ocr_utils import get_adjusted_mfdetrec_res
    import rapid_doc.backend.pipeline.analyze_utils
    import rapid_doc.model.table.rapid_table as old
    from rapid_doc.model.table.rapid_table_self import ModelType
    plans=[];payloads=[]
    def plan(original,layout,*,scale=1,extract=True,rotation=None,formulas=None,checkboxes=None,image_enabled=True,formula_enabled=True,prior=None):
        page=dict(ori_image_list=copy.deepcopy(original),rotate_label=rotation,table_fill_image_list=copy.deepcopy(prior))
        table=dict(poly=[10,10,100,10,100,100,10,100],layout_image_list=copy.deepcopy(layout),unknown=2**54+1)
        useful=[3,7,11,17,160,128,160,128]
        request=dict(page=copy.deepcopy(page),table=copy.deepcopy(table),useful_list=useful,formulas=formulas or [],checkboxes=checkboxes or [],scale=scale,formula_enabled=formula_enabled,image_enabled=image_enabled,extract_original_image=extract)
        adjusted=get_adjusted_mfdetrec_res(copy.deepcopy(request['formulas']+request['checkboxes']),useful,return_text=True)if formula_enabled else[]
        fills=sp.extract_table_fill_image(page,dict(table_res=table,useful_list=useful),scale,extract and rotation not in ['90','180','270'])if image_enabled else[]
        plans.append(dict(request=request,expected=dict(page=page,table=table,formulas=adjusted,fill_images=fills)))
    layout=[dict(poly=[12,14,25,14,25,28,12,28],uuid='layout',extra='keep')]
    for bbox in [[10,10,100,100],[10,10,91,100],[10,10,math.nextafter(91,0),100],[10,10,math.nextafter(91,100),100],[11,11,99,99],[20,20,40,40],[0,0,2,2],[100,100,100,100],[10,10,100,10]]:
        for scale in [1,1.,1.37]:
            for rotation in [None,'90','180','270',90]:
                plan([dict(bbox=bbox,uuid='original',extra={'opaque_image_ref':'img'})],layout,scale=scale,rotation=rotation,prior=[dict(uuid='prior')])
    for originals in [[],None,[dict(bbox=[20,20,40,40],uuid='one'),dict(bbox=[50,50,80,80],uuid='two')]]:
        for lay in [None,[],layout+[dict(poly=[-10,-20,-1,-20,-1,-2,-10,-2],uuid='outside')]]:
            for extract in [False,True]:
                for enabled in [False,True]:plan(originals,lay,extract=extract,image_enabled=enabled)
    for bbox in [[0,0,1,1],[-100,-100,-1,-1],[11,17,171,145],[10,16,11,17],[171,145,172,146],[172,146,175,150],[20,30,10,5]]:
        for value in ['',None,'x<&>',0,False]:
            for enabled in [False,True]:plan([],[],formulas=[dict(bbox=bbox,latex=value,checkbox='✓',other='drop')],checkboxes=[dict(bbox=[50,60,70,80],checkbox='香')],formula_enabled=enabled)
    source='build/migration/table-text/pixels.png';rgb=np.asarray(Image.open(ROOT/source).convert('RGB'))
    def quad(b):return [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
    base=[[quad([1,1,9,9]),quad([10,10,30,30]),[[45,10],[70,12],[60,30],[40,28]]],['&amp;<keep>','香','raw'],[.5,0.,1]]
    images=[None,[],[dict(ocr_bbox=quad([0,0,10,10]),uuid='<raw&uuid>')],[dict(ocr_bbox=quad([0,0,40,40]),uuid='first'),dict(ocr_bbox=quad([0,0,40,40]),uuid='second')],[dict(ocr_bbox=quad([10.9,10.1,1.1,1.9]),uuid='reverse')],[dict(ocr_bbox=quad([-3.5,-2.9,1.9,2.1]),uuid='negative')],[dict(ocr_bbox=quad([159,127,200,300]),uuid='edge')]]
    formulas=[None,[],[dict(bbox=[10.123456789,-.0,40.123456789,30],latex=' x<&> ',checkbox='ignored')],[dict(bbox=[1,2,3,4],checkbox='香'),dict(bbox=[5,6,7,8],latex='',checkbox='哦樂')],[dict(bbox=[0,0,1,1],latex=''),dict(bbox=[0,0,1,1],latex=None)]]
    saved=(old.inline_left_delimiter,old.inline_right_delimiter)
    def payload(ocr,ims,ms,skip,delimiters):
        calls=[];ocr_calls=[]
        def capture(images,rows):calls.append(dict(image=pixel(images[0][:,:,::-1].copy()),ocr_result=norm(rows[0])));return SimpleNamespace(pred_htmls=[None])
        model=old.RapidTableModel.__new__(old.RapidTableModel);model.model_type=ModelType.SLANETPLUS;model.table_model=capture
        model.ocr_engine=SimpleNamespace(ocr=lambda *a,**k:(ocr_calls.append(True)or [[]]))
        old.inline_left_delimiter,old.inline_right_delimiter=delimiters
        request=dict(ocr_result=copy.deepcopy(ocr),fill_images=copy.deepcopy(ims),formulas=copy.deepcopy(ms),skip_text_in_image=skip,inline_left=delimiters[0],inline_right=delimiters[1])
        model.predict(rgb.copy(),ocr_result=copy.deepcopy(ocr),fill_image_res=copy.deepcopy(ims),mfd_res=copy.deepcopy(ms),skip_text_in_image=skip,skip_table_orientation=True)
        expected=dict(ocr_result=calls[0]['ocr_result']if calls else None,needs_ocr=bool(ocr_calls))
        payloads.append(dict(source=source,source_sha256=digest((ROOT/source).read_bytes()),request=request,expected=expected,image=calls[0]['image']if calls else pixel(rgb)))
    try:
        for ocr in [None,[],[[],[],[]],base]:
            for ims in images:
                for ms in formulas:
                    for skip in [False,True]:payload(ocr,ims,ms,skip,('$','$'))
        for d in [(r'\(',r'\)'),('<&','>'),('','')]:payload(base,images[2],formulas[2],True,d)
        rng=random.Random(118)
        for i in range(60):
            rect=[rng.uniform(-50,200)for _ in range(4)]
            payload(base,[dict(ocr_bbox=quad(rect),uuid=str(i))],formulas[2],bool(i%2),('$','$'))
    finally:old.inline_left_delimiter,old.inline_right_delimiter=saved
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/ocr_utils.py','rapid_doc/utils/boxbase.py','rapid_doc/model/table/rapid_table.py','rapid_doc/model/table/utils.py']
    write_json(target,dict(plans=plans,payloads=payloads,sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual legacy adjusted formulas/image selection and actual RapidTableModel.predict before structural inference. Table model captures inputs only; empty OCR branch fixed to no detections, represented as needs_ocr=True, not falsely executed by Rust. Images are normalized JSON metadata with external uuid assets, no cross-call Python identity alias guarantee.'))
    print(f'Frozen {len(plans)} hint plans/{len(payloads)} actual model payload and exact pixel cases')
if __name__=='__main__':main()
