"""导出当前页面方向模型 RGB 路径的张量与分类，不验证模型实际识别准确率。"""
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.orientation.rapid_orientation.utils import ResizeImage,CropImage,NormalizeImage,ToCHWImage
from common import ROOT,write_json,digest

def prepare(image):
    for op in [ResizeImage(resize_short=256),CropImage(224),NormalizeImage(),ToCHWImage()]:image=op(image)
    return np.array(image)[None]

def main():
    model=ROOT/"rapid_doc/resources/rapid_orientation.onnx";options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    labels=session.get_modelmeta().custom_metadata_map["character"].splitlines()
    rng=np.random.default_rng(632);cases=[];pngs=[]
    for height,width in [(3,5),(77,43),(256,256),(512,640),(224,1000)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8);tensor=prepare(image)
        cases.append(dict(id=f"{width}x{height}",width=width,height=height,rgb=image.ravel().tolist(),shape=list(tensor.shape),values=tensor.ravel().tolist()))
    path=ROOT/"build/migration/detection-images/text.png";image=Image.open(path).convert("RGB")
    for angle in [0,90,180,270]:
        rotated=image.rotate(angle,expand=True);path=ROOT/f"build/migration/detection-images/orientation-{angle}.png";rotated.save(path)
        result=session.run(None,{session.get_inputs()[0].name:prepare(np.asarray(rotated))})[0][0];index=int(np.argmax(result))
        pngs.append(dict(input=str(path),rotation_degrees=int(labels[index]),score=float(result[index])))
    write_json(ROOT/"build/migration/orientation-baseline.json",dict(model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes()),cases=cases,png_cases=pngs,scope="RGB path used by RapidOrientationModel.predict; no automatic orientation heuristic or rotation applied"))
    print(f"Exported {len(cases)} orientation tensors and {len(pngs)} PNG model classifications")
if __name__=="__main__":main()
