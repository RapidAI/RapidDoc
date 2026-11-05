"""导出 RapidDoc 配置下的文字检测前处理和原始概率图合同。"""

import numpy as np
import onnxruntime as ort
from rapidocr.ch_ppocr_det.utils import DetPreProcess
from common import ROOT,digest,write_json


def main():
    """保持 RapidDoc 的固定 limit_side_len，不使用上游动态边长选择。"""
    rng=np.random.default_rng(456)
    model=ROOT/"rapid_doc/resources/PP-OCRv6_det_small.onnx"
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    cases=[]
    for height,width,side in [(40,80,64),(77,43,64),(96,640,128),(129,257,128),(97,641,128),(32,32,64),(31,65,64),(48,320,320),(1,10,64)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        for mode in ("min","max"):
            tensor=DetPreProcess(side,mode,[0.485,0.456,0.406],[0.229,0.224,0.225])(image)
            value={"id":f"{width}x{height}-{mode}","width":width,"height":height,"side":side,"mode":mode,"bgr":image.ravel().tolist()}
            if tensor is None:
                value["expected_error"]="invalid_input"
            else:
                value.update(shape=list(tensor.shape),values=tensor.ravel().tolist())
                if (height,width)==(40,80):
                    outputs=session.run(None,{session.get_inputs()[0].name:tensor})
                    value["outputs"]={name.name:{"shape":list(result.shape),"values":result.ravel().tolist()}for name,result in zip(session.get_outputs(),outputs)}
            cases.append(value)
    write_json(ROOT/"build/migration/detection-preprocess.json",{"model":model.relative_to(ROOT).as_posix(),"sha256":digest(model.read_bytes()),"cases":cases,"scope":"BGR resize/normalization, original FP32 probability maps; contour/DB boxes not covered"})
    print(f"Exported {len(cases)} detector size/mode cases and 2 CPU probability-map baselines")


if __name__=="__main__":main()
