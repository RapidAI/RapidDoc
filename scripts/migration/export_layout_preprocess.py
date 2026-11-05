"""冻结 PP-DocLayoutV3 RGB cubic 前处理张量，不涉及框后处理。"""
import numpy as np
import cv2
from PIL import Image
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.model_handler.utils import ModelType
from common import ROOT,write_json,digest

def main():
    rng=np.random.default_rng(335);cases=[]
    for height,width,target in [(3,5,32),(77,43,64),(96,640,32),(97,641,64),(32,32,32),(1,10,32)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        tensor=PPPreProcess((target,target),ModelType.PP_DOCLAYOUTV3)(image)
        cases.append(dict(id=f"{width}x{height}-{target}",width=width,height=height,target=target,rgb=image.ravel().tolist(),shape=list(tensor.shape),values=tensor.ravel().tolist()))
    image=np.asarray(Image.open(ROOT/"build/migration/detection-images/text.png").convert("RGB"));tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(image)
    cases.append(dict(id="text-800",width=image.shape[1],height=image.shape[0],target=800,rgb=image.ravel().tolist(),shape=list(tensor.shape),values=tensor.ravel().tolist()))
    # 大张量以位模式哈希冻结，避免把大量十进制浮点数重复写入 JSON。
    output=ROOT/"build/migration/layout-inputs";output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(444)
    for height,width,target in [(4,4,32),(4,7,32),(7,4,64),(17,33,64),(43,77,800),(137,641,800),(320,480,800),(1024,768,800)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        tensor=PPPreProcess((target,target),ModelType.PP_DOCLAYOUTV3)(image)
        path=output/f"{width}x{height}-{target}.rgb";path.write_bytes(image.tobytes())
        cases.append(dict(id=f"extra-{width}x{height}-{target}",width=width,height=height,target=target,rgb_file=path.relative_to(ROOT).as_posix(),shape=list(tensor.shape),fp32_le_sha256=digest(tensor.astype("<f4",copy=False).tobytes())))
    write_json(ROOT/"build/migration/layout-preprocess.json",dict(cases=cases,opencv=cv2.__version__,ipp_enabled=cv2.ipp.useIPP(),ipp_version=cv2.ipp.getIppVersion()))
    print(f"Exported {len(cases)} layout RGB cubic tensors")
if __name__=="__main__":main()
