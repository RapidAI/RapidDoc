"""冻结实际旧原生检测框、native阶段模型和OCR检测输入；检测固定空响应只隔离输入规划。"""
import copy,json,math,random
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/native-ocr-plan.json'
    if target.exists():raise FileExistsError(target)
    from pdftext.schema import Bbox
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc.utils.span_pre_proc import txt_spans_bbox_extract
    from rapid_doc.utils.ocr_utils import get_adjusted_mfdetrec_res
    from rapid_doc.utils.model_utils import crop_img
    def normalize(v):
        if isinstance(v,Bbox):return normalize(v.bbox)
        if isinstance(v,np.ndarray):return normalize(v.tolist())
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:normalize(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [normalize(x)for x in v]
        return v
    def adapt(page):
        page=copy.deepcopy(page)
        for b in page['blocks']:
            for line in b['lines']:
                for s in line['spans']:s['bbox']=Bbox(s['bbox'])
        return page
    def span(text='a',box=(10.,10.,20.,20.),size=10):return dict(text=text,bbox=list(box),font=dict(size=size))
    def page(spans=None,rotation=0.):return dict(blocks=[dict(lines=[dict(rotation=rotation,spans=spans or [])])])
    def region(box=(0.,0.,100.,100.)):
        x,y,r,b=box;return dict(category_id=1,poly=[x,y,r,y,r,b,x,b],score=.95,original_label='text',original_order=1)
    boxes=[]
    def add(spans=None,rotation=0.,scale=1.0,masks=None,useful=None,res=None,native=None):
        request=dict(native_page=native or page(spans,rotation),region=res or region(),scale=scale,masks=masks or [],useful_list=useful or [50,50,0,0,100,100,200,200])
        r=copy.deepcopy(request['region']);actual=txt_spans_bbox_extract(adapt(request['native_page']),r,copy.deepcopy(request['masks']),scale,request['useful_list'])
        boxes.append(dict(request=request,expected=normalize(dict(region=r,boxes=actual))))
    add();add([span()]);add([span()],scale=1)
    for text in ['a','.','(','abc','',')','“','—']:
        for x in [-5.,0.,5.,95.,100.]:add([span(text,(x,10.,x+10.,20.))])
    for angle in [0.,math.pi/2,math.pi,math.pi*1.5,math.radians(.099999),math.radians(.1),math.radians(.100001),math.radians(45),-math.pi/2]:add([span()],rotation=angle)
    for gap in [5.999999999999999,6.,6.000000000000001,10.]:
        for size in [0,10,20]:add([span(box=(10.,10.,20.,20.)),span(box=(20.+gap,10.,30.+gap,20.),size=size)])
    for y in [17.999999999999996,18.,18.000000000000004]:add([span(),span(box=(25.,y,35.,y+10))])
    for mask in [dict(bbox=[62,60,65,70]),dict(bbox=[62.,60.,65.,70.]),dict(bbox=[0,0,1,1]),dict(bbox=[-100,60,1000,70]),dict(bbox=[62,60.1,65,68.1])]:
        for scale in [1,1.0,2.5]:add([span()],scale=scale,masks=[mask])
    r=random.Random(719)
    for _ in range(100):
        spans=[span(r.choice(['a','.',')','']),[float(r.randrange(90)),float(r.randrange(90)),0.,0.],r.choice([6,10,16]))for _ in range(r.randrange(1,25))]
        for s in spans:s['bbox'][2]=s['bbox'][0]+r.randrange(1,20);s['bbox'][3]=s['bbox'][1]+r.randrange(1,15)
        add(spans,scale=r.choice([1,1.,.5,2.5]),masks=[dict(bbox=[55.,50.,62.5,150.])]if r.randrange(2)else[])
    # Freeze full native stage + actual old grouped/padded detector inputs on real prior crop plans.
    base=json.loads((ROOT/'build/migration/legacy-model.json').read_text());plans=[]
    for i,c in enumerate(base['crops'][:18]):
        for ocr_enable,mode in [(False,'auto'),(True,'auto'),(False,'ocr')]:
            model=copy.deepcopy(c['expected']['model']);tasks=copy.deepcopy(c['expected']['ocr']);np_img=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
            for t in tasks:
                t['source']=t['source']
            facts=page([span('Native',(5.,10.,40.,20.)),span('Next',(6.,24.,35.,33.))])
            if i%4==0:facts=page()
            if i%7==0:facts['rotate_label']='90'
            request=dict(plan=dict(model=copy.deepcopy(model),ocr=tasks),native_page=copy.deepcopy(facts),scale=1.,ocr_enable=ocr_enable,use_det_mode=mode)
            dets=model['layout_dets'];formulas=[d for d in dets if d['category_id']in[8,13,14]];checkboxes=[d for d in dets if d['category_id']==200]
            entry=dict(ocr_res_list=[dets[t['layout_index']]for t in tasks],layout_res=dets,np_img=np_img,page_idx=0,lang=None,ocr_enable=ocr_enable,single_page_mfdetrec_res=formulas,checkbox_res=checkboxes)
            observed=[];captured=[];original_boxes=old.txt_spans_bbox_extract;original_manager=old.AtomModelSingleton
            def observe(pd,res,mfd_res,scale,useful_list):
                value=original_boxes(pd,res,mfd_res,scale,useful_list);idx=next(i for i,d in enumerate(dets)if d is res);observed.append(dict(layout_index=idx,boxes=normalize(value)));return value
            class Detector:
                def det_batch_predict(self,images,batch_size):
                    for image in images:captured.append(dict(width=image.shape[1],height=image.shape[0],pixels_sha256=digest(image[:,:,::-1].copy().tobytes())))
                    return [(np.empty((0,4,2),np.float32),0.)for _ in images]
            class Manager:
                def get_atom_model(self,**kwargs):return Detector()
            try:
                old.txt_spans_bbox_extract=observe;old.AtomModelSingleton=Manager
                if mode!='ocr':old._extract_text_from_pdf([entry],[adapt(facts)],[1.])
                pending=[]
                for t in tasks:
                    res=dets[t['layout_index']];enabled=entry['ocr_enable']
                    if not enabled:
                        if res.get('need_ocr_det'):enabled=True
                        elif mode=='txt'or(mode!='ocr'and not res.get('need_ocr_det')):continue
                    _,u=crop_img(res,np_img,crop_paste_x=50,crop_paste_y=50)
                    masks=get_adjusted_mfdetrec_res(formulas+checkboxes,u)
                    pending.append(dict(t,masks=normalize(masks),ocr_enable=enabled))
                old._run_ocr_det_batch([entry],Manager(),dict(use_det_mode=mode))
            finally:old.txt_spans_bbox_extract=original_boxes;old.AtomModelSingleton=original_manager
            groups={}
            for t in pending:
                u=t['useful_list'];groups.setdefault((((u[7]+63)//64)*64,((u[6]+63)//64)*64),[]).append(t)
            detect=[]
            for shape,group in groups.items():
                for t in group:
                    t=copy.deepcopy(t);t['crop_source']=t['source'];t['source']=f'detect-3-{len(detect)}';t['target_shape']=list(shape);detect.append(t)
            assert len(captured)==len(detect)
            expected=dict(model=model,native=observed,detect=detect)
            plans.append(dict(request=request,crop_case=i,expected=normalize(expected),images=[dict(key=t['source'],**image)for t,image in zip(detect,captured)]))
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/boxbase.py','rapid_doc/utils/ocr_utils.py','rapid_doc/backend/pipeline/analyze_utils.py']
    write_json(target,dict(boxes=boxes,plans=plans,sources={p:digest((ROOT/p).read_bytes())for p in refs},profile='实际旧原生框和native模型；仅bbox对象投影数组，旧临时span字段副作用不写回调用方。实际_run_ocr_det_batch固定空检测响应，仅隔离并捕获送入模型的真实BGR补白/遮罩数组，RGB反转后另比原像素，不宣称执行实际检测模型。'))
    print(f'Frozen {len(boxes)} native bbox cases, {len(plans)} actual native/detection-input plans/{sum(len(p["images"])for p in plans)} detector input pixel refs')

if __name__=='__main__':main()
