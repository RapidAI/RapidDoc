"""UNET 原始类别图回缩、矩形闭运算及有序线连接的分层合同。"""
import cv2,json,math,numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.table.rapid_table_self.table_structure.unet.main import TSRUnetStructurer
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_line_rec import get_table_line,adjust_lines,final_adjust_lines
from common import ROOT,write_json,digest

def capture(name,mask,width,height):
    cases=[]
    for morph_close in [False,True]:
        h=mask.copy();h[h==2]=0;v=mask.copy();v[v==1]=0
        h=cv2.resize(h,(width,height));v=cv2.resize(v,(width,height))
        vk=cv2.getStructuringElement(cv2.MORPH_RECT,(1,int(math.sqrt(mask.shape[0])*1.2)))
        hk=cv2.getStructuringElement(cv2.MORPH_RECT,(int(math.sqrt(mask.shape[1])*1.2),1))
        v=cv2.morphologyEx(v,cv2.MORPH_CLOSE,vk)
        if morph_close:h=cv2.morphologyEx(h,cv2.MORPH_CLOSE,hk)
        rows=get_table_line(h,axis=0,lineW=50);columns=get_table_line(v,axis=1,lineW=30)
        rows+=adjust_lines(rows,alph=100,angle=50);columns+=adjust_lines(columns,alph=15,angle=50)
        rows,columns=final_adjust_lines(rows,columns)
        r=np.asarray(rows,np.float32).reshape(-1,4);c=np.asarray(columns,np.float32).reshape(-1,4)
        cases.append(dict(id=f"{name}-{morph_close}",width=mask.shape[1],height=mask.shape[0],mask=mask.ravel().tolist(),source_width=width,source_height=height,morph_close=morph_close,masks_sha256=digest(h.tobytes()+v.tobytes()),lines_sha256=digest(r.astype("<f4").tobytes()+c.astype("<f4").tobytes()),rows=r.tolist(),columns=c.tolist()))
    return cases

def main():
    cases=[];rng=np.random.default_rng(672)
    for w,h,tw,th in [(1,1,1,1),(64,48,96,72),(64,48,31,17),(64,48,64,48),(37,53,173,121),(160,120,480,240),(1024,512,480,240)]:
        mask=np.zeros((h,w),np.uint8)
        for y in [h//5,h//2,h*4//5]:mask[y:y+2,:]=1
        for x in [w//5,w//2,w*4//5]:mask[:,x:x+2]=2
        # 可见间隙，验证偶数 anchor 的方向和垂直总是 close 的行为。
        mask[h//2,w//2:w//2+3]=0
        cases+=capture(f"synthetic-{w}x{h}-{tw}x{th}",mask,tw,th)
    fixture=json.loads((ROOT/"build/migration/unet-baseline.json").read_text())
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/fixture["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    pre=TSRUnetStructurer.__new__(TSRUnetStructurer);pre.inp_height=pre.inp_width=1024
    pre.mean=np.array([123.675,116.28,103.53],np.float32);pre.std=np.array([58.395,57.12,57.375],np.float32)
    for case in fixture["png_cases"]:
        image=np.asarray(Image.open(case["input"]).convert("RGB"))[:,:,::-1].copy()
        mask=session.run(None,{"input":pre.preprocess(image)["img"]})[0][0,0].astype(np.uint8)
        cases+=capture(case["input"],mask,image.shape[1],image.shape[0])
    write_json(ROOT/"build/migration/unet-lines.json",dict(cases=cases))
    print(f"Exported {len(cases)} UNET resized/closed masks and line geometry contracts")

if __name__=="__main__":main()
