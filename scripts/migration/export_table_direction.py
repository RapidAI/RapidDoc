"""冻结实际旧表格方向投票/分类门槛/旋转与_process_single_table方向链；不覆盖旧参考。"""
import argparse,copy,json,math
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

def pixel(a):return dict(width=a.shape[1],height=a.shape[0],pixels_sha256=digest(a.tobytes()))
def adapt(page):
    page=copy.deepcopy(page)
    for b in page['blocks']:
        for line in b['lines']:
            for s in line['spans']:s['bbox']=SimpleNamespace(bbox=s['bbox'])
    return page
def quad(w,h):return [[1.,1.],[1.+w,1.],[1.+w,1.+h],[1.,1.+h]]
def page(angle=0,rotation=0,text='A',bbox=(20,20,40,40),rotate_label=None):
    return dict(rotation=rotation,rotate_label=rotate_label,blocks=[dict(lines=[dict(rotation=angle,spans=[dict(bbox=list(bbox),text=text)])])])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--portrait',action='store_true');parser.add_argument('--empty-portrait',action='store_true');args=parser.parse_args()
    target=ROOT/('build/migration/table-direction-empty-portrait.json'if args.empty_portrait else 'build/migration/table-direction-portrait.json'if args.portrait else 'build/migration/table-direction.json')
    if target.exists():raise FileExistsError(target)
    import rapid_doc.utils.span_pre_proc as sp
    import rapid_doc.backend.pipeline.analyze_utils as analyze
    from rapid_doc.utils.boxbase import rotate_table_image
    from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
    from rapid_doc.model.orientation.rapid_orientation import RapidOrientation
    from rapid_doc.backend.pipeline.model_list import AtomicModel
    table=dict(poly=[10,10,100,10,100,100,10,100]);plans=[]
    def plan(pg,width=100,height=130,boxes=None,scale=1.):
        boxes=boxes or []
        label,angles=sp.txt_most_angle_extract_table(adapt(pg),dict(table_res=copy.deepcopy(table)),scale)if pg.get('rotate_label')not in ('90','180','270')else('0',[])
        model=RapidOrientationModel.__new__(RapidOrientationModel);calls=[]
        model.orientation_engine=lambda im:(calls.append(True)or '0',0.)
        if not angles:model.predict(np.zeros((height,width,3),np.uint8),copy.deepcopy(boxes))
        request=dict(page=copy.deepcopy(pg),table=copy.deepcopy(table),width=width,height=height,boxes=copy.deepcopy(boxes),scale=scale)
        plans.append(dict(request=request,expected=dict(rotate_label=label,angles=angles,needs_classifier=bool(calls),pdf_not_rotate=pg.get('rotate_label')not in ('90','180','270'))))
    for angle in [0,math.pi/2,math.pi,3*math.pi/2,-math.pi/2,math.radians(.5),math.radians(1.5),math.radians(89.5),math.radians(90.5),math.radians(-.5),math.radians(-1.5),math.radians(45),math.radians(720.5),'bad','NaN','inf',None,False,True,' 1.5707963267948966 ',[],{}]:
        for pr in [0,90,-90,'270','bad',None]:plan(page(angle,pr))
    for text in ['A','',' \t\n','\x1c\u3000','.',')','（','"','A.','[','—']:
        for bbox in [(20,20,40,40),(90,20,130,40),(-20,20,20,40),(0,0,20,20),(100,20,120,40)]:plan(page(math.pi/2,text=text,bbox=bbox))
    for label in ['90','180','270',90,None,'0','unknown']:
        for scale in [.5,1.,2.,3.7]:plan(page(math.pi/2,rotate_label=label),scale=scale)
    for rotations in [[90,0],[270,90,270,90],[45,90,90],[0,270],[180,90,180]]:
        pg=dict(blocks=[dict(lines=[dict(rotation=math.radians(a),spans=[dict(text='A',bbox=[20,20,30,30])])for a in rotations])]);plan(pg)
    for height in [119,120,121,200]:
        for n in [0,1,2,3,10,11,14,25]:
            for vertical in sorted(set([0,min(n,2),min(n,3),n])):
                qs=[quad(1,10)]*vertical+[quad(10,1)]*(n-vertical)
                plan(dict(blocks=[]),height=height,boxes=qs)
    for w in [math.nextafter(.8,0),.8,math.nextafter(.8,1),-1.]:
        plan(dict(blocks=[]),boxes=[quad(w,1)]*3)
    source=ROOT/'build/migration/table-text/pixels.png';rgb=np.asarray(Image.open(source).convert('RGB'));pixels=[]
    if args.portrait or args.empty_portrait:
        source=ROOT/('build/migration/table-direction/color-portrait.png'if args.empty_portrait else 'build/migration/table-direction/portrait.png');source.parent.mkdir(parents=True,exist_ok=True)
        if source.exists():raise FileExistsError(source)
        rgb=np.rot90(rgb).copy();Image.fromarray(rgb).save(source)
    for label in ['0','90','180','270','45']:
        task=dict(table_img=rgb.copy());rotate_table_image(task,label)
        pixels.append(dict(source=source.relative_to(ROOT).as_posix(),angle=label,classifier_bgr=False,expected=pixel(task['table_img'])))
    pixels.append(dict(source=source.relative_to(ROOT).as_posix(),angle='0',classifier_bgr=True,expected=pixel(rgb[:,:,::-1].copy())))
    # Actual full legacy function with detector/OCR/structure fixed only to expose orientation execution.
    saved=(analyze._detect_table_text,analyze._run_table_ocr)
    workflows=[]
    for native in [None,0,90,180,270,45]:
        for classifier in ['0','90','180','270']:
            initial=[quad(1,10)]*3
            pg=page(math.radians(native))if native is not None else dict(blocks=[])
            trace=[];classifier_inputs=[];final=[]
            model=RapidOrientationModel.__new__(RapidOrientationModel)
            model.orientation_engine=lambda im:(classifier_inputs.append(pixel(im.copy()))or classifier,0.)
            def detect(worker,im,masks,**kw):trace.append(dict(image=pixel(im[:,:,::-1].copy()),masks=copy.deepcopy(masks)));return copy.deepcopy(initial if len(trace)==1 else [quad(12,3)])
            analyze._detect_table_text=detect
            analyze._run_table_ocr=lambda worker,im,boxes,*a:(final.append(dict(image=pixel(im[:,:,::-1].copy()),boxes=copy.deepcopy(boxes)))or [])
            tm=SimpleNamespace(predict=lambda *a,**k:None)
            manager=SimpleNamespace(get_atom_model=lambda **k:model if k['atom_model_name']==AtomicModel.ImgOrientationCls else tm)
            task=dict(table_res=copy.deepcopy(table),table_img=rgb.copy(),lang=None,useful_list=[0,0,0,0,160,128,160,128],single_page_mfdetrec_res=[],checkbox_res=[],ocr_enable=True)
            try:analyze._process_single_table(task,adapt(pg),1.,manager,dict(use_rule_table=False,force_ocr=True,table_formula_enable=False,table_image_enable=False),{})
            finally:analyze._detect_table_text,analyze._run_table_ocr=saved
            workflows.append(dict(source=source.relative_to(ROOT).as_posix(),page=pg,table=table,initial=initial,second=[quad(12,3)],classifier_label=classifier,trace=trace,classifier_inputs=classifier_inputs,final=final))
    # Real detector + real orientation model through actual legacy orchestration; capture boxes at OCR boundary.
    import onnxruntime as ort
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapid_doc.model.orientation.rapid_orientation.utils import ResizeImage,CropImage,NormalizeImage,ToCHWImage
    prior=json.loads((ROOT/'build/migration/table-text.json').read_text(encoding='utf-8'))
    ori=ROOT/'rapid_doc/resources/rapid_orientation.onnx';opts=ort.SessionOptions();opts.intra_op_num_threads=1
    ds=ort.InferenceSession(str(ROOT/prior['det_model']),sess_options=opts,providers=['CPUExecutionProvider']);os=ort.InferenceSession(str(ori),sess_options=opts,providers=['CPUExecutionProvider'])
    det=TextDetector.__new__(TextDetector);det.limit_type='max';det.limit_side_len=960;det.mean=[.485,.456,.406];det.std=[.229,.224,.225];det.session=lambda t:ds.run(None,{ds.get_inputs()[0].name:t})[0];det.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)
    worker=RapidOcrModel.__new__(RapidOcrModel);worker.text_detector=det;worker.is_seal=False;worker.enable_merge_det_boxes=False
    engine=RapidOrientation.__new__(RapidOrientation);engine.session=lambda t:os.run(None,{os.get_inputs()[0].name:t});engine.labels=os.get_modelmeta().custom_metadata_map['character'].splitlines();engine.preprocess_ops=[ResizeImage(resize_short=256),CropImage(224),NormalizeImage(),ToCHWImage()]
    real=[]
    if args.empty_portrait:
        blank=ROOT/'build/migration/table-direction/blank-portrait.png'
        if blank.exists():raise FileExistsError(blank)
        Image.new('RGB',(128,160),'white').save(blank)
    for name in (['blank-portrait','color-portrait']if args.empty_portrait else ['orientation-90','orientation-270']if args.portrait else ['text','rotated','blank']):
        path=ROOT/f'build/migration/{"table-direction"if args.empty_portrait else "detection-images"}/{name}.png';arr=np.asarray(Image.open(path).convert('RGB'))
        for native in [None,90,180,270]:
            pg=page(math.radians(native))if native is not None else dict(blocks=[])
            trace=[];used=[];angles=[];labels=[];retries=[];final=[]
            def real_detect(w,im,masks,**kw):
                first=worker.ocr(im,mfd_res=masks,rec=False)[0]
                result=saved[0](worker,im,masks,**kw);trace.append(dict(image=pixel(im[:,:,::-1].copy()),boxes=result or []));retries.append(not bool(first)and min(im.shape[:2])<32);return result
            model=RapidOrientationModel.__new__(RapidOrientationModel)
            model.orientation_engine=lambda im:(used.append(pixel(im.copy()))or engine(im))
            original_predict=model.predict
            model.predict=lambda *a,**k:(labels.append(original_predict(*a,**k))or labels[-1])
            tm=SimpleNamespace(predict=lambda *a,**k:None)
            manager=SimpleNamespace(get_atom_model=lambda **k:model if k['atom_model_name']==AtomicModel.ImgOrientationCls else worker if k['atom_model_name']==AtomicModel.OCR else tm)
            task=dict(table_res=copy.deepcopy(table),table_img=arr.copy(),lang=None,useful_list=[0,0,0,0,arr.shape[1],arr.shape[0],arr.shape[1],arr.shape[0]],single_page_mfdetrec_res=[],checkbox_res=[],ocr_enable=True)
            analyze._detect_table_text=real_detect;analyze._run_table_ocr=lambda w,im,boxes,*a:(final.append(boxes)or [])
            try:analyze._process_single_table(task,adapt(pg),1.,manager,dict(use_rule_table=False,force_ocr=True,table_formula_enable=False,table_image_enable=False),{})
            finally:analyze._detect_table_text,analyze._run_table_ocr=saved
            native_label,aa=sp.txt_most_angle_extract_table(adapt(pg),dict(table_res=table),1.)
            real.append(dict(source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path.read_bytes()),page=pg,table=table,expected=dict(rotate_label=native_label if aa else labels[-1],angles=aa,pdf_not_rotate=True,classifier_used=bool(used),retries=retries,boxes=trace[-1]['boxes']),image=pixel(task['table_img']),trace=trace,classifier_inputs=used))
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/boxbase.py','rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/orientation/rapid_orientation_model.py','rapid_doc/model/orientation/rapid_orientation/rapid_orientation.py']
    write_json(target,dict(plans=plans,pixels=pixels,workflows=workflows,real=real,sources={p:digest((ROOT/p).read_bytes())for p in refs},orientation_model=ori.relative_to(ROOT).as_posix(),orientation_sha256=digest(ori.read_bytes()),det_model=prior['det_model'],det_sha256=prior['det_sha256'],scope='Actual legacy native angle selection, orientation call gating, rotations and detection orchestration. Fixed-response workflows isolate policy. Real CPU detection/orientation only; recognition and structure model not run.'))
    print(f'Frozen {len(plans)} plans/{len(pixels)} pixel rotations/{len(workflows)} old workflows/{len(real)} actual CPU chains; classifier invoked {sum(c["expected"]["classifier_used"]for c in real)} times')
if __name__=='__main__':main()
