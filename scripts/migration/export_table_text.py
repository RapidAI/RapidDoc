"""冻结实际旧表格行级OCR、薄图重试和_process_single_table模型结果回填；固定响应仅隔离策略。"""
import copy,json,math,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,digest,write_json

def norm(v):
    if isinstance(v,np.ndarray):return norm(v.tolist())
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:norm(x)for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [norm(x)for x in v]
    return v
def pixel(a):return dict(width=a.shape[1],height=a.shape[0],pixels_sha256=digest(a.tobytes()))
def main():
    target=ROOT/'build/migration/table-text.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.utils.ocr_utils as old
    import rapid_doc.backend.pipeline.analyze_utils as analyze
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    folder=target.with_suffix('');folder.mkdir(parents=True,exist_ok=True)
    rgb=np.fromfunction(lambda y,x,c:(x*17+y*31+c*67)%256,(128,160,3),dtype=int).astype(np.uint8)
    source=folder/'pixels.png';Image.fromarray(rgb).save(source)
    masks_cases=[[],[dict(bbox=[20.2,5.9,41.1,50.01])],[dict(bbox=[-2,-3,2,2])],[dict(bbox=[0,0,999,999])],[dict(bbox=[9,9,1,1])]]
    masks=[]
    for ms in masks_cases:
        masked=analyze._apply_mask_boxes_to_image(rgb[:,:,::-1].copy(),copy.deepcopy(ms))[:,:,::-1].copy()
        masks.append(dict(source=source.relative_to(ROOT).as_posix(),request=dict(masks=ms),expected=pixel(masked),padded=pixel(np.pad(masked,((8,8),(8,8),(0,0)),constant_values=255))))
    boxes=[]
    prior=json.loads((ROOT/'build/migration/model-ocr-execution.json').read_text(encoding='utf-8'))
    for c in prior['cases']:
        request=c['request'];q=old.sorted_boxes(np.asarray(request['boxes'],np.float32))
        if request['masks']:q=old.update_det_boxes(q,copy.deepcopy(request['masks']))
        boxes.append(dict(request=request,expected=norm(q)))
    restores=[]
    for width,height in [(1,1),(8,31),(31,32),(160,16),(1,128)]:
        for padding in [1,8,32]:
            for value in [0.,math.nextafter(8.,0),8.,math.nextafter(8.,99),999.]:
                q=[[[value,-2.1],[value+13.12,2.2],[value+13.12,9.9],[value,9.9]]]
                class Fake:
                    count=0
                    def ocr(self,img,**kwargs):self.count+=1;return [None if self.count==1 else q]
                # force retry threshold above both dimensions; actual function doesF32 then clip.
                out=analyze._detect_table_text(Fake(),np.zeros((height,width,3),np.uint8),[],retry_padding=padding,retry_short_side=1000)
                restores.append(dict(request=dict(boxes=q,width=width,height=height,padding=padding),expected=out))
    applies=[];crops=[]
    def quad(x,y,w,h):return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
    qs=[quad(1,1,1,12),quad(2,5,35,10),quad(5,3,10,20),[[2,2],[39,4],[38,14],[1,12]]]
    texts=[None,'',' 香 ','哦樂','1號','0號','10號','第6號','１號','&amp;<x>"\'','\x1c  A  B \u3000',' α ', ' \n 中文 \t ']
    for i,q in enumerate(qs):
        im=old.get_rotate_crop_image(rgb[:,:,::-1].copy(),np.asarray(q,np.float32))[:,:,::-1].copy();crops.append(dict(boxes=[q],source=source.relative_to(ROOT).as_posix(),expected=[pixel(im)]))
        for text in texts:
            for score in [0.,.49999,.5,1.00001]:
                class Fake:
                    def ocr(self,images,**kwargs):return [[(text,score)]]
                expected=analyze._run_table_ocr(Fake(),rgb[:,:,::-1].copy(),[q],False)
                applies.append(dict(request=dict(boxes=[q],results=[[text,score]]),expected=norm(expected)))
    applies.append(dict(request=dict(boxes=[],results=[]),expected=[]))
    # Actual final _process_single_table branch, earlier stages fixed to empty OCR/known images.
    attachments=[]
    saved=(analyze._detect_table_text,analyze.txt_most_angle_extract_table,analyze.extract_table_fill_image)
    htmls=[None,'','x','<TABLE>x</TABLE>','<table class="x">x</table>','before<table>中文</table>after','</table> x <table>','a<table><table>x</table></table>z']
    try:
        analyze._detect_table_text=lambda *a,**k:[]
        analyze.txt_most_angle_extract_table=lambda *a,**k:('0',[0])
        for html in htmls:
            for scale in [.5,1.,2.,3.7]:
                for populated in [False,True]:
                    formulas=[dict(bbox=[-3.5,2.9,50.1,60.9]),dict(bbox=[999,999,1001,1001])]if populated else[]
                    fill=[dict(bbox=[1,2,3,4],ori_bbox=[-3.9,6.1,10.5,19.9]),dict(ori_bbox=[9,9,10,10])]if populated else[]
                    table=dict(category_id=5,poly=[0,0,80,0,80,90,0,90],layout_image_list=['remove'],html='prior',unknown=2**54+1,formula_boxes=['prior'],img_boxes=['prior'])
                    request=dict(table=copy.deepcopy(table),html_code=html,formulas=formulas,fill_images=fill,scale=scale)
                    analyze.extract_table_fill_image=lambda *a,**k:copy.deepcopy(fill)
                    model=SimpleNamespace(predict=lambda *a,**k:html)
                    manager=SimpleNamespace(get_atom_model=lambda **k:model)
                    task=dict(table_res=table,table_img=rgb,lang=None,useful_list=[0,0,0,0,160,128,160,128],single_page_mfdetrec_res=formulas,checkbox_res=[],ocr_enable=True)
                    analyze._process_single_table(task,{},scale,manager,dict(use_rule_table=False,force_ocr=True,table_formula_enable=False),{})
                    attachments.append(dict(request=request,expected=norm(table)))
    finally:analyze._detect_table_text,analyze.txt_most_angle_extract_table,analyze.extract_table_fill_image=saved
    # Real patched RapidOCR DB/CTC, default table unclip1.6 and no merge, actual _detect/_run.
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1
    det_path=ROOT/prior['det_model'];rec_path=ROOT/prior['rec_model']
    det_session=ort.InferenceSession(str(det_path),sess_options=options,providers=['CPUExecutionProvider']);rec_session=ort.InferenceSession(str(rec_path),sess_options=options,providers=['CPUExecutionProvider'])
    detector=TextDetector.__new__(TextDetector);detector.limit_type='max';detector.limit_side_len=960;detector.mean=[.485,.456,.406];detector.std=[.229,.224,.225]
    detector.session=lambda t:det_session.run(None,{det_session.get_inputs()[0].name:t})[0]
    detector.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)
    worker=RapidOcrModel.__new__(RapidOcrModel);worker.text_detector=detector;worker.is_seal=False;worker.enable_merge_det_boxes=False
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    worker.text_recognizer=SimpleNamespace(rec_batch_num=6,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=lambda t:rec_session.run(None,{rec_session.get_inputs()[0].name:t})[0],postprocess_op=CTCLabelDecode(character=rec_session.get_modelmeta().custom_metadata_map['character'].splitlines()))
    paths=[ROOT/f'build/migration/detection-images/{name}.png'for name in ['text','rotated','blank']]
    im=Image.new('RGB',(320,12),'white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);ImageDraw.Draw(im).text((8,-8),'Table A 1 B 2',font=font,fill='black');p=folder/'thin.png';im.save(p);paths.append(p)
    real=[]
    for path in paths:
        with Image.open(path)as im:arr=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
        for ms in [[],[dict(bbox=[0,0,arr.shape[1]//3,arr.shape[0]])]]:
            masked=analyze._apply_mask_boxes_to_image(arr,ms);first=worker.ocr(masked,mfd_res=ms,rec=False)[0]
            detected=analyze._detect_table_text(worker,masked,ms)
            result=analyze._run_table_ocr(worker,arr,detected,False)if detected else[]
            real.append(dict(source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path.read_bytes()),masks=ms,expected=dict(boxes=norm(detected or []),ocr_result=norm(result),retried=not bool(first)and min(arr.shape[:2])<32)))
    refs=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/ocr/rapid_ocr.py','rapid_doc/model/ocr/ocr_patch.py','rapid_doc/utils/ocr_utils.py','rapid_doc/model/table/utils.py']
    write_json(target,dict(masks=masks,boxes=boxes,restores=restores,applies=applies,crops=crops,attachments=attachments,real=real,sources={p:digest((ROOT/p).read_bytes())for p in refs},det_model=prior['det_model'],det_sha256=prior['det_sha256'],rec_model=prior['rec_model'],rec_sha256=prior['rec_sha256'],scope='Actual table line OCR and model-result attachment only. Fixed responses isolate policy; real CPU detection/recognition listed separately. No native route/orientation/native text/structure model orchestration.'))
    print(f'Frozen {len(boxes)} boxes/{len(restores)} retries/{len(applies)} OCR policies/{len(attachments)} HTML attachments and {len(real)} real CPU chains')
if __name__=='__main__':main()
