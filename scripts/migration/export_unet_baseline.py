"""UNET 前处理位模式与原始 int64 类别图的 CPU 基线。"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.table.rapid_table_self.table_structure.unet.main import TSRUnetStructurer
from common import ROOT,write_json,digest

def main():
    pre=TSRUnetStructurer.__new__(TSRUnetStructurer)
    pre.inp_height=pre.inp_width=1024
    pre.mean=np.array([123.675,116.28,103.53],np.float32)
    pre.std=np.array([58.395,57.12,57.375],np.float32)
    rng=np.random.default_rng(501);cases=[]
    directory=ROOT/"build/migration/unet-inputs";directory.mkdir(parents=True,exist_ok=True)
    for w,h in [(3,5),(480,240),(77,43),(641,137),(1024,1024),(1600,1200),(2048,2048),
                (3,17),(17,3),(333,250),(800,650),(1050,1027),(3072,1536),(4096,2048)]:
        image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        path=directory/f"{w}x{h}.bgr";path.write_bytes(image.tobytes())
        tensor=pre.preprocess(image)["img"]
        expected=directory/f"{w}x{h}.fp32";expected.write_bytes(tensor.astype("<f4",copy=False).tobytes())
        cases.append(dict(input=str(path),width=w,height=h,shape=list(tensor.shape),expected=str(expected),fp32_le_sha256=digest(expected.read_bytes())))
    path=ROOT/"rapid_doc/model/table/rapid_table_self/models/unet.onnx"
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=["CPUExecutionProvider"])
    table=json.loads((ROOT/"build/migration/table-structure.json").read_text(encoding="utf-8"));pngs=[]
    for case in table["models"][0]["png_cases"]:
        image=np.asarray(Image.open(case["input"]).convert("RGB"))[:,:,::-1].copy();tensor=pre.preprocess(image)["img"]
        mask=session.run(None,{"input":tensor})[0][0,0].astype(np.uint8)
        pngs.append(dict(input=case["input"],width=mask.shape[1],height=mask.shape[0],sha256=digest(mask.tobytes())))
    write_json(ROOT/"build/migration/unet-baseline.json",dict(cases=cases,png_cases=pngs,model=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes())))
    print(f"Exported {len(cases)} UNET input tensors and {len(pngs)} raw line masks")

if __name__=="__main__":main()
