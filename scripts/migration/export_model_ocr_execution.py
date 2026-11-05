"""冻结实际旧框整理/模型级rec回填/裁图及真实CPU det→rec→回填链。"""
import copy,json,math,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/model-ocr-execution.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.utils.ocr_utils as old
    import rapid_doc.backend.pipeline.analyze_utils as analyze
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
    import onnxruntime as ort
    def norm(v):
        if isinstance(v,np.ndarray):return norm(v.tolist())
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:norm(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [norm(x)for x in v]
        return v
    def rect(x,y,w,h):return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    def post(boxes,masks):
        q=old.merge_det_boxes(old.sorted_boxes(np.asarray(boxes,np.float32)))if len(boxes)else[]
        if len(q)and masks:q=old.update_det_boxes(q,copy.deepcopy(masks))
        return q
    cases=[]
    def add(q,masks=()):cases.append(dict(request=dict(boxes=q,masks=list(masks)),expected=norm(post(q,masks))))
    add([])
    for dy in [5.9999995,6.,6.0000005,9.999999,10.,10.000001]:
        for w in [39.999996,40.,40.000004]:add([rect(50,10,w,10),rect(0,10+dy,w,10)])
    for ratio in [.6,.8]:
        for shift in [math.nextafter(10*(1-ratio),0),10*(1-ratio),math.nextafter(10*(1-ratio),10)]:
            add([rect(0,0,100,10)], [dict(bbox=[20,shift,30,10+shift])])
    for tilt in [0.,.01,1.,2.,3.,5.]:
        add([[[5,5],[65,5+tilt],[65,15+tilt],[5,15]],rect(10,40,30,10)])
    r=random.Random(845)
    for _ in range(100):
        q=[]
        for _ in range(r.randrange(1,22)):
            x,y=r.randrange(100),r.randrange(80);q.append(rect(x,y,r.randrange(2,40),r.randrange(2,20)))
        add(q,[dict(bbox=[30,0,40,128])]if r.randrange(2)else[])
    folder=ROOT/'build/migration/model-ocr-execution';folder.mkdir(parents=True,exist_ok=True)
    rgb=np.fromfunction(lambda y,x,k:(x*17+y*31+k*67)%256,(128,160,3),dtype=int).astype(np.uint8)
    source=folder/'pixels.png';Image.fromarray(rgb).save(source)
    preparations=[]
    def prepare(q,masks,enabled,source_path=source,u=None):
        with Image.open(source_path)as im:arr=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
        u=u or [50,50,10,20,10+arr.shape[1],20+arr.shape[0],arr.shape[1],arr.shape[0]]
        model=dict(layout_dets=[dict(category_id=1,poly=[10.,20.,150.,20.,150.,100.,10.,100.],original_label='text',original_order=1,score=.95)])
        job=dict(layout_index=0,source='det',crop_source='src',masks=masks,ocr_enable=enabled,useful_list=u,target_shape=[arr.shape[0],arr.shape[1]])
        q_final=post(q,masks);py_boxes=[v.tolist()if hasattr(v,'tolist')else v for v in q_final]
        items=old.get_ocr_result_list(py_boxes,u,enabled,arr,None,'text',1)
        images=[];rec=[]
        for i,item in enumerate(items):
            if 'np_img'in item:
                a=item['np_img'];key=f'rec-4-{len(rec)}';images.append(dict(key=key,width=a.shape[1],height=a.shape[0],pixels_sha256=digest(a[:,:,::-1].copy().tobytes())))
                item['np_img']=key;rec.append(dict(source=key,model_index=i+1))
        model['layout_dets'].extend(items)
        expected=dict(model=model,processed=[dict(layout_index=0,boxes=norm(q_final),ocr_enable=enabled or old.is_mostly_tilted(py_boxes))],recognition=rec)
        request=dict(plan=dict(model=dict(layout_dets=copy.deepcopy(model['layout_dets'][:1])),detect=[job]),detections=[q],lang=None,page_index=4)
        return dict(request=request,source=source_path.relative_to(ROOT).as_posix(),expected=norm(expected),images=images)
    for q in [[],[rect(5,5,30,10)],[rect(5,5,10,15)],[rect(5,5,10,19.9)],[rect(5,5,10,20)],[rect(5,5,2,10)],[[[5,5],[65,10],[65,25],[5,20]]], [rect(5,5,40,10),rect(40,5,30,10)]]:
        for enabled in [False,True]:
            for masks in [[],[dict(bbox=[25,0,29,100])]]:preparations.append(prepare(q,masks,enabled))
    applies=[]
    for text in ['ok','号','（204号','（20','（2','（2号','（20号','（204']:
        for score in [.49999999999999994,.5,.5000000000000001,.7999999999999999,.8,.8000000000000002,.5005]:
            for vertical in [False,True]:
                model=dict(layout_dets=[dict(category_id=15,poly=[0.,0.,10. if vertical else 40.,0.,10. if vertical else 40.,40.,0.,40.],text='',score=1,np_img='r',lang=None,unknown=True)])
                actual=copy.deepcopy(model);actual['layout_dets'][0]['np_img']=np.zeros((10,10,3),np.uint8)
                class Worker:
                    def ocr(self,*args,**kwargs):return [(text,score)],None
                class Manager:
                    def get_atom_model(self,**kwargs):return Worker()
                previous=analyze.AtomModelSingleton
                try:analyze.AtomModelSingleton=Manager;analyze._run_ocr_rec_postprocess([actual['layout_dets']],{})
                finally:analyze.AtomModelSingleton=previous
                applies.append(dict(request=dict(model=model,results=[[text,score]]),expected=norm(actual)))
    # Real CPU old batch-size1 detector and batch-size6 recognizer, no fixed response.
    det_path=ROOT/'rapid_doc/resources/PP-OCRv6_det_small.onnx';rec_path=ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx'
    options=ort.SessionOptions();options.intra_op_num_threads=1
    det_session=ort.InferenceSession(str(det_path),sess_options=options,providers=['CPUExecutionProvider']);rec_session=ort.InferenceSession(str(rec_path),sess_options=options,providers=['CPUExecutionProvider'])
    detector=TextDetector.__new__(TextDetector);detector.limit_type='max';detector.limit_side_len=960;detector.mean=[.485,.456,.406];detector.std=[.229,.224,.225]
    detector.session=lambda t:det_session.run(None,{det_session.get_inputs()[0].name:t})[0]
    detector.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.8,use_dilation=True)
    worker=RapidOcrModel.__new__(RapidOcrModel);worker.text_detector=detector
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    worker.text_recognizer=SimpleNamespace(rec_batch_num=6,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=lambda t:rec_session.run(None,{rec_session.get_inputs()[0].name:t})[0],postprocess_op=CTCLabelDecode(character=rec_session.get_modelmeta().custom_metadata_map['character'].splitlines()))
    real=[]
    paths=[ROOT/f'build/migration/detection-images/{name}.png'for name in ['text','rotated','blank']]
    for width in [1280,2048]:
        im=Image.new('RGB',(width,256),'white');font=ImageFont.truetype(str(ROOT/'rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf'),32);ImageDraw.Draw(im).text((30,30),'RapidDoc real pipeline 123',fill='black',font=font);p=folder/f'wide-{width}.png';im.save(p);paths.append(p)
    for path in paths:
        with Image.open(path)as im:arr=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
        detected=worker.det_batch_predict([arr],1)[0][0]
        q=norm(detected);p=prepare(q,[],True,path,[0,0,0,0,arr.shape[1],arr.shape[0],arr.shape[1],arr.shape[0]])
        p['detected']=q;p['source_sha256']=digest(path.read_bytes())
        actual=copy.deepcopy(p['expected']['model']);items=old.get_ocr_result_list([v.tolist()for v in post(q,[])],p['request']['plan']['detect'][0]['useful_list'],True,arr,None,'text',1)
        actual['layout_dets']=actual['layout_dets'][:1]+items
        class Manager:
            def get_atom_model(self,**kwargs):return worker
        previous=analyze.AtomModelSingleton
        try:analyze.AtomModelSingleton=Manager;analyze._run_ocr_rec_postprocess([actual['layout_dets']],{})
        finally:analyze.AtomModelSingleton=previous
        p['after_recognition']=norm(actual);real.append(p)
    refs=['rapid_doc/utils/ocr_utils.py','rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/ocr/rapid_ocr.py']
    write_json(target,dict(cases=cases,preparations=preparations,applies=applies,real=real,det_model=det_path.relative_to(ROOT).as_posix(),det_sha256=digest(det_path.read_bytes()),rec_model=rec_path.relative_to(ROOT).as_posix(),rec_sha256=digest(rec_path.read_bytes()),sources={p:digest((ROOT/p).read_bytes())for p in refs},profile='实际旧sorted/merge/update/get_ocr_result_list；np_img换ID并另比原RGB，未改其他字段。固定识别响应仅隔离阈值，5组真实CPU det批1→rec批6→旧回填另列；无错误重试等价声明。'))
    print(f'Frozen {len(cases)} box rules, {len(preparations)} recognition plans/{sum(len(c["images"])for c in preparations)} crop pixels, {len(applies)} rec policies and {len(real)} actual CPU chains')

if __name__=='__main__':main()
