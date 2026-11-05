"""用已有识别模型导出中英文、空白和批量的 CTC 差分输入。"""

import importlib.metadata
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from common import ROOT,digest,write_json


def main():
    """图像原本就是 48x320，不掩盖尚未移植的 resize/padding 行为。"""
    model=ROOT/"rapid_doc/resources/PP-OCRv6_rec_small.onnx"
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    dictionary=session.get_modelmeta().custom_metadata_map["character"].splitlines()
    decoder=CTCLabelDecode(character=dictionary)
    font=ImageFont.truetype(str(ROOT/"rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf"),28)
    tensors=[];cases=[]
    for index,text in enumerate(["RapidDoc 123","中文识别测试","","bookkeeper 2026"]):
        image=Image.new("RGB",(320,48),"white");ImageDraw.Draw(image).text((4,4),text,font=font,fill="black")
        tensor=np.asarray(image)[:,:,::-1].astype(np.float32).transpose(2,0,1)/255.
        tensor=(tensor-0.5)/0.5;tensors.append(tensor)
    for index,tensor in enumerate(tensors):cases.append((f"line-{index}",tensor[np.newaxis]))
    cases.append(("batch-two",np.stack(tensors[:2])))
    result=[]
    for name,tensor in cases:
        predictions=session.run(None,{session.get_inputs()[0].name:tensor})[0]
        expected=decoder(predictions)[0]
        indices=predictions.argmax(axis=2)
        outputs=[]
        for row,(text,confidence) in zip(indices,expected):
            selected=np.ones(len(row),dtype=bool);selected[1:]=row[1:]!=row[:-1];selected&=row!=0
            outputs.append({"text":text,"confidence":confidence,"token_indices":row[selected].tolist(),"timesteps":np.where(selected)[0].tolist()})
        result.append({"id":name,"shape":list(tensor.shape),"values":tensor.ravel().tolist(),"expected":outputs})
        print(name,[(item["text"],item["confidence"])for item in outputs])
    write_json(ROOT/"build/migration/ctc-fixtures.json",{"model":model.relative_to(ROOT).as_posix(),"sha256":digest(model.read_bytes()),"python_ort":ort.__version__,"rapidocr":importlib.metadata.version("rapidocr"),"scope":"CTC and normalized original-size BGR lines only; resize/padding, detection, word boxes and bidi are not covered","cases":result})


if __name__=="__main__":main()
