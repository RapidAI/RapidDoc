"""冻结实际PP-FormulaNet旧预处理像素和FP32张量；不运行公式网络。"""
import json,random
import numpy as np
import PIL,cv2
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/formula-preprocess.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.pre_process import PPPreProcess
    folder=target.with_suffix('');folder.mkdir(parents=True,exist_ok=True)
    cases=[];rng=random.Random(7074)
    def add(rgb,target_size):
        i=len(cases);source=folder/f'{i}.png';Image.fromarray(rgb).save(source)
        old=PPPreProcess(target_size)
        cropped=old.uni_mer_net_img_decode.crop_margin(Image.fromarray(rgb))
        decoded=old.uni_mer_net_img_decode.img_decode(rgb)
        tensor=old([rgb])[0]
        decoded.tofile(folder/f'{i}.rgb');tensor.astype('<f4').tofile(folder/f'{i}.f32')
        # bbox由非白边内容反查不可靠，单独冻结旧crop_margin对坐标颜色图的裁切大小。
        cases.append(dict(source=str(source.relative_to(ROOT)).replace('\\','/'),target_size=list(target_size),input_size=[rgb.shape[1],rgb.shape[0]],cropped_size=list(cropped.size),shape=list(tensor.shape),rgb=str((folder/f'{i}.rgb').relative_to(ROOT)).replace('\\','/'),tensor=str((folder/f'{i}.f32').relative_to(ROOT)).replace('\\','/'),rgb_sha256=digest(decoded.tobytes()),tensor_sha256=digest(tensor.astype('<f4').tobytes())))
    for color in [0,1,127,128,254,255]:
        add(np.full((17,31,3),color,np.uint8),(31,47))
    for w,h in [(1,1),(1,257),(257,1),(13,1701),(1701,13),(37,47),(81,11),(11,81),(3,19),(19,3),(91,73)]:
        rgb=np.fromfunction(lambda y,x,c:(x*17+y*29+c*71)%256,(h,w,3),dtype=int).astype(np.uint8)
        add(rgb,(16,32));add(rgb,(35,29))
    for _ in range(50):
        w,h=rng.randrange(2,201),rng.randrange(2,161)
        rgb=np.random.default_rng(rng.randrange(100000)).integers(0,256,(h,w,3),dtype=np.uint8)
        add(rgb,(rng.randrange(2,97),rng.randrange(2,97)))
    for size in [(384,384),(768,768)]:
        for w,h in [(100,100),(91,37),(257,33),(133,19),(11,81)]:
            rgb=np.full((h,w,3),255,np.uint8)
            rgb[2:-2,3:-3]=np.fromfunction(lambda y,x,c:(x*31+y*37+c*79)%256,(h-4,w-6,3),dtype=int).astype(np.uint8)
            add(rgb,size)
    paths=['rapid_doc/model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/pre_process.py']
    write_json(target,dict(schema=1,scope='actual old preprocessing only; no formula inference',versions=dict(pillow=PIL.__version__,numpy=np.__version__,opencv=cv2.__version__),references={p:digest((ROOT/p).read_bytes())for p in paths},cases=cases))
    print(f'Frozen {len(cases)} exact RGB/tensor formula preprocessing cases')

if __name__=='__main__':main()
