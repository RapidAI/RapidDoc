"""UNET 显式有线裁图的专用 matcher/空格补 rec/HTML CPU 基线。"""
import cv2,copy,html,json
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr.ch_ppocr_det.utils import DetPreProcess,DBPostProcess
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.utils.process_img import get_rotate_crop_image
from rapid_doc.model.table.rapid_table_self.table_structure.unet.unet_table_rec import UnetTableRecognition
from rapid_doc.model.table.utils import normalize_table_ocr_text
from rapid_doc.utils.span_pre_proc import calculate_contrast
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import plot_html_table
from export_wired_match import capture,ccw
from common import ROOT,write_json,digest

def blank_crop(image,p):
    x1,y1=int(p[0][0])+1,int(p[0][1])+1;x2,y2=int(p[2][0])-1,int(p[2][1])-1
    if x1>=x2 or y1>=y2 or x1<0 or y1<0 or (x2-x1)/max(y2-y1,1)>20 or (y2-y1)/max(x2-x1,1)>20:return None
    crop=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)[y1:y2,x1:x2]
    if not crop.size:return None
    return crop,float(calculate_contrast(crop,'bgr'))

def main():
    options=ort.SessionOptions();options.intra_op_num_threads=1
    detector_path=ROOT/'rapid_doc/resources/PP-OCRv6_det_small.onnx';recognizer_path=ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx'
    detector=ort.InferenceSession(str(detector_path),sess_options=options,providers=['CPUExecutionProvider'])
    session=ort.InferenceSession(str(recognizer_path),sess_options=options,providers=['CPUExecutionProvider'])
    decoder=CTCLabelDecode(character=session.get_modelmeta().custom_metadata_map['character'].splitlines())
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    def recognize(crop):
        tensor=rec.resize_norm_img(crop,max(320/48,crop.shape[1]/crop.shape[0]))[None]
        return decoder(session.run(None,{session.get_inputs()[0].name:tensor})[0])[0][0]
    def blanks(image,polygons,needed):
        results=[]
        for i,p in enumerate(polygons):
            if not needed[i]:continue
            result=dict(cell_index=i,text='',confidence=0.,accepted=False,reason='invalid_crop',contrast=None)
            crop=blank_crop(image,p)
            if crop is not None:
                crop,contrast=crop;result['contrast']=contrast
                if contrast<=.17:result['reason']='low_contrast'
                else:
                    text,score=recognize(crop);accept=not UnetTableRecognition._should_drop_blank_cell_rec_result(text,score)
                    result.update(text=html.unescape(normalize_table_ocr_text(text))if accept else text,confidence=float(score),accepted=accept,reason='accepted'if accept else'filtered')
            results.append(result)
        return results
    rng=np.random.default_rng(963);image=rng.integers(0,256,(64,96,3),np.uint8);crop_cases=[]
    for i,b in enumerate([[0,0,96,64],[-2,-2,20,20],[2,2,95,5],[20,10,130,80],[20,20,20,30],[1.9,2.9,50.9,40.9]]):
        p=ccw(b);value=blank_crop(image,p)
        crop_cases.append(dict(id=f'crop-{i}',bgr=image.ravel().tolist(),width=96,height=64,points=p,expected=None if value is None else dict(width=value[0].shape[1],height=value[0].shape[0],sha256=digest(value[0].tobytes()),contrast=value[1])))
    regions=json.loads((ROOT/'build/migration/table-regions.json').read_text());cases=[]
    for case in regions['png_cases']:
        image=np.asarray(Image.open(case['input']).convert('RGB'))[:,:,::-1].copy()
        tensor=DetPreProcess(960,'max',[.485,.456,.406],[.229,.224,.225])(image)
        pred=detector.run(None,{detector.get_inputs()[0].name:tensor})[0]
        boxes,_=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)(pred,image.shape[:2])
        groups=[]
        for box in sorted(boxes,key=lambda b:b[0][1]):
            if not groups or abs(box[0][1]-groups[-1][-1][0][1])>=10:groups.append([])
            groups[-1].append(box)
        boxes=[b for group in groups for b in sorted(group,key=lambda b:b[0][0])];lines=[]
        for box in boxes:
            text,score=recognize(get_rotate_crop_image(image,np.array(box,np.float32)))
            lines.append(dict(points=box.tolist(),text=text,confidence=float(score)))
        matched=capture(case['input'],case['polygons'],lines)['expected']
        needed=[not cell for cell in matched['cells']];blank=blanks(image,case['polygons'],needed)
        final=copy.deepcopy(matched['cell_lines'])
        for b in blank:
            if b['accepted']:final[b['cell_index']]=[b['text'].replace('\r','').replace('\n','').strip()]
        escape=lambda s:s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace("'",'&#39;')
        rendered=plot_html_table(case['logic_points']or[],{i:[escape(t)for t in values]for i,values in enumerate(final)},case['polygons'])
        # 强制全部格补 rec，验证检测失败时仍能直接识别已有文字的物理格。
        all_blank=blanks(image,case['polygons'],[True]*len(case['polygons']))
        cases.append(dict(input=case['input'],polygons=case['polygons'],logic_points=case['logic_points'],ocr_lines=lines,match=matched,blank_rec=blank,all_blank_rec=all_blank,cell_lines=final,html=rendered))
    write_json(ROOT/'build/migration/wired-pipeline.json',dict(cases=cases,crop_cases=crop_cases,unet_model=regions['model'],unet_sha256=regions['sha256'],det_model=detector_path.relative_to(ROOT).as_posix(),det_sha256=digest(detector_path.read_bytes()),rec_model=recognizer_path.relative_to(ROOT).as_posix(),rec_sha256=digest(recognizer_path.read_bytes())))
    print(f'Exported {len(crop_cases)} color crop/contrast contracts and {len(cases)} wired PNG pipelines')

if __name__=='__main__':main()
