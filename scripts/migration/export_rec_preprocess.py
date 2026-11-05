"""导出当前 RapidOCR/OpenCV 识别前处理和真实 PNG 行的基线。"""

from pathlib import Path
import importlib.metadata
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from common import ROOT,digest,write_json


def main():
    """仅调用现有纯前处理方法，不构造 RapidOCR 会话或触发模型下载。"""
    recognizer=TextRecognizer.__new__(TextRecognizer);recognizer.rec_image_shape=[3,48,320]
    rng=np.random.default_rng(123)
    result=[]
    for height,width in [(3,5),(12,17),(40,99),(96,640),(1,10),(48,320),(64,777),(97,641),(48,1000)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        tensor=recognizer.resize_norm_img(image,max(320/48,width/height))
        result.append({"id":f"random-{width}x{height}","width":width,"height":height,"bgr":image.ravel().tolist(),"shape":[1,*tensor.shape],"values":tensor.ravel().tolist()})
    model=ROOT/"rapid_doc/resources/PP-OCRv6_rec_small.onnx"
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    decoder=CTCLabelDecode(character=session.get_modelmeta().custom_metadata_map["character"].splitlines())
    output=ROOT/"build/migration/recognition-lines";output.mkdir(parents=True,exist_ok=True)
    pngs=[]
    for index,(text,width,height,size)in enumerate([("RapidDoc 123",300,40,28),("中文识别测试",240,64,38),("",320,48,28),("bookkeeper 2026",480,72,44)]):
        image=Image.new("RGB",(width,height),"white")
        font=ImageFont.truetype(str(ROOT/"rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf"),size)
        ImageDraw.Draw(image).text((4,4),text,font=font,fill="black")
        path=output/f"line-{index}.png";image.save(path)
        bgr=np.asarray(image)[:,:,::-1].copy()
        tensor=recognizer.resize_norm_img(bgr,max(320/48,width/height))[np.newaxis]
        predictions=session.run(None,{session.get_inputs()[0].name:tensor})[0]
        expected=decoder(predictions)[0]
        pngs.append({"input":str(path),"text":expected[0][0],"confidence":expected[0][1]})
        result.append({"id":f"png-{index}","width":width,"height":height,"bgr":bgr.ravel().tolist(),"shape":list(tensor.shape),"values":tensor.ravel().tolist()})
        print(path.name,expected)
    write_json(ROOT/"build/migration/recognition-preprocess.json",{"opencv":cv2.__version__,"rapidocr":importlib.metadata.version("rapidocr"),"model":model.relative_to(ROOT).as_posix(),"sha256":digest(model.read_bytes()),"tensor_cases":result,"png_cases":pngs})


if __name__=="__main__":main()
