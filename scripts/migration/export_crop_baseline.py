"""冻结透视裁图像素，以及真实检测框内的逐行识别基线。"""
import json
import numpy as np
import onnxruntime as ort
from rapidocr.utils.process_img import get_rotate_crop_image
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from common import ROOT,write_json,digest

def main():
    rng=np.random.default_rng(69);image=rng.integers(0,256,(48,64,3),dtype=np.uint8)
    cases=[]
    for name,points in [
        ("axis",[[4,3],[44,3],[44,23],[4,23]]),("vertical",[[3,2],[13,2],[13,42],[3,42]]),
        ("sloped",[[9,3],[55,12],[50,36],[4,27]]),("perspective",[[9,3],[55,12],[48,40],[4,27]]),
        ("border",[[-4,-2],[44,1],[45,23],[-4,25]]),("tiny",[[0,0],[3,0],[3,2],[0,2]])]:
        crop=get_rotate_crop_image(image,np.array(points,np.float32))
        cases.append(dict(id=name,width=64,height=48,bgr=image.ravel().tolist(),points=points,crop=dict(width=crop.shape[1],height=crop.shape[0],bgr=crop.ravel().tolist())))
    fixture=json.loads((ROOT/"build/migration/db-baseline.json").read_text(encoding="utf-8"))
    model=ROOT/"rapid_doc/resources/PP-OCRv6_rec_small.onnx"
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=opts,providers=["CPUExecutionProvider"])
    decoder=CTCLabelDecode(character=session.get_modelmeta().custom_metadata_map["character"].splitlines())
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    pngs=[]
    for value in fixture["png_cases"]:
        bgr=np.asarray(Image.open(value["input"]).convert("RGB"))[:,:,::-1].copy();results=[]
        for index,points in enumerate(value["boxes"]):
            points=np.array(points,np.float32);crop=get_rotate_crop_image(bgr,points)
            tensor=rec.resize_norm_img(crop,max(320/48,crop.shape[1]/crop.shape[0]))[None]
            result=decoder(session.run(None,{session.get_inputs()[0].name:tensor})[0])[0][0]
            results.append(dict(text=result[0],confidence=result[1]))
        pngs.append(dict(input=value["input"],boxes=value["boxes"],recognitions=results))
    pdf=ROOT/"build/migration/detection-images/scanned.pdf"
    document=canvas.Canvas(str(pdf),pagesize=(320,128),invariant=1)
    for value in fixture["png_cases"]:
        document.drawImage(ImageReader(value["input"]),0,0,width=320,height=128);document.showPage()
    document.save()
    hybrid=pdf.with_name("hybrid.pdf")
    document=canvas.Canvas(str(hybrid),pagesize=(320,128),invariant=1)
    document.drawString(20,80,"Native text page");document.showPage()
    document.drawImage(ImageReader(fixture["png_cases"][0]["input"]),0,0,width=320,height=128);document.showPage()
    document.showPage();document.save()
    write_json(ROOT/"build/migration/crop-baseline.json",dict(cases=cases,png_cases=pngs,pdf=str(pdf),hybrid_pdf=str(hybrid),model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes())))
    print(f"Exported {len(cases)} crop pixel cases, {sum(len(p['boxes'])for p in pngs)} real region recognitions")
if __name__=="__main__":main()
