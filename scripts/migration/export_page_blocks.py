"""冻结页面正文裁图和旧 CPU DB/CTC 参考；不导入 Rust 扩展。"""
import json,math
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from PIL import Image
from rapidocr.ch_ppocr_det.utils import DetPreProcess,DBPostProcess
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.utils.process_img import get_rotate_crop_image
from common import ROOT,write_json,digest

def make_ocr():
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
    detector=ort.InferenceSession(str(ROOT/'rapid_doc/resources/PP-OCRv6_det_small.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    recognizer=ort.InferenceSession(str(ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    decoder=CTCLabelDecode(character=recognizer.get_modelmeta().custom_metadata_map['character'].splitlines())
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    def ocr(image):
        bgr=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        tensor=DetPreProcess(960,'max',[.485,.456,.406],[.229,.224,.225])(bgr)
        boxes,_=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)(detector.run(None,{detector.get_inputs()[0].name:tensor})[0],bgr.shape[:2])
        groups=[]
        for box in sorted(boxes,key=lambda b:b[0][1]):
            if not groups or abs(box[0][1]-groups[-1][-1][0][1])>=10:groups.append([])
            groups[-1].append(box)
        lines=[]
        for box in [b for group in groups for b in sorted(group,key=lambda b:b[0][0])]:
            crop=get_rotate_crop_image(bgr,np.array(box,np.float32))
            tensor=rec.resize_norm_img(crop,max(320/48,crop.shape[1]/crop.shape[0]))[None]
            text,score=decoder(recognizer.run(None,{recognizer.get_inputs()[0].name:tensor})[0])[0][0]
            lines.append(dict(points=box.tolist(),text=text,confidence=float(score)))
        return lines
    return ocr

def main():
    source=json.loads((ROOT/'build/migration/page-tables.json').read_text())
    output=ROOT/'build/migration/page-blocks';output.mkdir(parents=True,exist_ok=True)
    ocr=make_ocr()
    def capture(name,path,layout,synthetic=False):
        image=Image.open(path).convert('RGB');regions=[]
        for index,region in enumerate(layout):
            if region['label']not in ('text','doc_title','paragraph_title'):continue
            x0,y0,x1,y1=region['coordinate']
            bounds=[max(0,math.floor(x0/5)*5),max(0,math.floor(y0/5)*5),min(image.width,math.ceil(x1/5)*5),min(image.height,math.ceil(y1/5)*5)]
            crop=image.crop(bounds);target=output/f'{name}-{index}.png';crop.save(target)
            regions.append(dict(layout_index=index,bounds=bounds,input=str(target),rgb_sha256=digest(crop.tobytes()),lines=ocr(crop)))
        return dict(input=path,layout=layout,text_regions=regions,synthetic=synthetic)
    cases=[capture(Path(c['input']).stem,c['input'],c['layout'])for c in source['cases']]
    path=source['cases'][0]['input']
    # 刻意打乱数组/模型顺序；同 order 时保持原数组顺序，未知类别必须保留。
    values=[('text',[55,85,500,125],7),('doc_title',[55,40,450,80],2),('display_formula',[10,750,30,780],3),('image',[20,790,70,840],7)]
    layout=[dict(cls_id=i,label=label,score=.95,coordinate=bounds,order=order)for i,(label,bounds,order)in enumerate(values)]
    cases.append(capture('synthetic',path,layout,True))
    assert sum(len(c['text_regions'])for c in cases)>=3
    assert any(line['text']for c in cases for r in c['text_regions']for line in r['lines'])
    write_json(ROOT/'build/migration/page-blocks.json',dict(cases=cases))
    print(f'Exported {len(cases)} page layouts with {sum(len(c["text_regions"])for c in cases)} independent CPU text-region references')

if __name__=='__main__':main()
