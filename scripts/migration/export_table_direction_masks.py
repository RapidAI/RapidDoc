"""补充实际旧_process_single_table旋转后沿用公式遮罩坐标；独立冻结，不覆写既有参考。"""
import copy,math
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/table-direction-masks.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc.utils.ocr_utils import get_adjusted_mfdetrec_res
    source=ROOT/'build/migration/table-direction/portrait.png';rgb=np.asarray(Image.open(source).convert('RGB'))
    def pixels(im):return dict(width=im.shape[1],height=im.shape[0],pixels_sha256=digest(im.tobytes()))
    cases=[];saved=(old._detect_table_text,old._run_table_ocr)
    for angle in [90,270]:
        for box in [[2.5,3.9,40.1,80.2],[-3,-2,129,161]]:
            formula=[dict(bbox=box,latex='x')];useful=[0,0,0,0,128,160,128,160]
            masks=get_adjusted_mfdetrec_res(formula,useful,return_text=True);trace=[]
            def detect(w,im,ms,**kw):trace.append(dict(image=pixels(im[:,:,::-1].copy()),masks=copy.deepcopy(ms)));return []
            old._detect_table_text=detect;old._run_table_ocr=lambda *a:[]
            table=dict(poly=[0,0,100,0,100,100,0,100])
            pg=dict(blocks=[dict(lines=[dict(rotation=math.radians(angle),spans=[dict(text='A',bbox=SimpleNamespace(bbox=[20,20,40,40]))])])])
            page=dict(blocks=[dict(lines=[dict(rotation=math.radians(angle),spans=[dict(text='A',bbox=[20,20,40,40])])])])
            task=dict(table_res=copy.deepcopy(table),table_img=rgb.copy(),lang=None,useful_list=useful,single_page_mfdetrec_res=formula,checkbox_res=[],ocr_enable=True)
            model=SimpleNamespace(predict=lambda *a,**k:None);manager=SimpleNamespace(get_atom_model=lambda **k:model)
            try:old._process_single_table(task,pg,1.,manager,dict(use_rule_table=False,force_ocr=True,table_image_enable=False),{})
            finally:old._detect_table_text,old._run_table_ocr=saved
            cases.append(dict(source=source.relative_to(ROOT).as_posix(),page=page,table=table,masks=masks,trace=trace,rotate_label=str(angle),image=pixels(task['table_img'])))
    refs=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/utils/ocr_utils.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual legacy masked detection inputs before/after native 90/270 direction. Fixed empty detection and structure isolate pixel/mask policy; no actual model inference.'))
    print(f'Frozen {len(cases)} actual legacy rotated mask workflows')
if __name__=='__main__':main()
