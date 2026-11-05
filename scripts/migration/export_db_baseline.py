"""保存 DB 独立概率图合同，避免轮廓偏差与模型数值偏差混杂。"""
import cv2
import numpy as np
from rapidocr.ch_ppocr_det.utils import DBPostProcess,DetPreProcess
from common import ROOT,write_json,digest
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont

def main():
    cases=[]
    maps=[]
    for angle in [0,15,30,60,-20]:
        value=np.zeros((96,160),np.float32)
        points=cv2.boxPoints(((80,48),(75,16),angle)).astype(np.int32)
        cv2.fillPoly(value,[points],0.9)
        maps.append((f"rectangle-{angle}",value))
    value=np.zeros((96,160),np.float32);value[0:20,0:80]=0.9;value[40:60,90:160]=0.8
    maps.append(("edges",value))
    value=np.zeros((96,160),np.float32);value[10:70,10:140]=0.9;value[30:50,40:100]=0
    maps.append(("hole",value))
    maps.append(("blank",np.zeros((96,160),np.float32)))
    rng=np.random.default_rng(128)
    maps.append(("noise",rng.uniform(0,1,(48,64)).astype(np.float32)))
    model=ROOT/"rapid_doc/resources/PP-OCRv6_det_small.onnx"
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    pngs=[]
    output=ROOT/"build/migration/detection-images";output.mkdir(parents=True,exist_ok=True)
    for name,angle in [("text",0),("rotated",15),("blank",0)]:
        image=Image.new("RGB",(640,256),"white")
        if name!="blank":
            draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(ROOT/"rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf"),32)
            draw.text((24,24),"RapidDoc 123",font=font,fill="black");draw.text((24,100),"中文识别测试",font=font,fill="black")
        if angle:image=image.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor="white")
        path=output/f"{name}.png";image.save(path)
        tensor=DetPreProcess(960,"max",[.485,.456,.406],[.229,.224,.225])(np.asarray(image)[:,:,::-1])
        prediction=session.run(None,{session.get_inputs()[0].name:tensor})[0]
        post=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)
        boxes,scores=post(prediction,(image.height,image.width))
        pngs.append(dict(input=str(path),boxes=boxes.tolist(),scores=scores))
        # 真实模型输出也必须走独立概率图差分。
        maps.append((f"model-{name}",prediction[0,0]))
    for name,value in maps:
        for dilation in [False,True]:
            post=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=dilation)
            boxes,scores=post(value[None,None],(value.shape[0],value.shape[1]))
            cases.append(dict(id=f"{name}-{dilation}",width=value.shape[1],height=value.shape[0],dilation=dilation,values=value.ravel().tolist(),boxes=boxes.tolist(),scores=scores))
    write_json(ROOT/"build/migration/db-baseline.json",dict(cases=cases,png_cases=pngs,model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes())))
    print(f"Exported {len(cases)} DB probability maps")
if __name__=="__main__":main()
