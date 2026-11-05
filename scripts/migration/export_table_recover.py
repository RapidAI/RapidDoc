"""UNET 四角框到逻辑格：冻结源索引、基准轨道和固定合并阈值。"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.table.rapid_table_self.table_structure.unet.main import TSRUnetStructurer
from rapid_doc.model.table.rapid_table_self.table_structure.unet.table_recover import TableRecover
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import sorted_ocr_boxes,box_4_2_poly_to_box_4_1
from common import ROOT,write_json

def polygon(x0,y0,x1,y1):return [[x0,y0],[x0,y1],[x1,y1],[x1,y0]]

def capture(name,polygons,row_threshold=10.,column_threshold=15.):
    # Python list 的公开合同是 float64；真实 float32 框保持每个数值，再用同一合同恢复。
    points=np.array(polygons,np.float64)
    _,logic=TableRecover()(points,row_threshold,column_threshold)
    result=dict(id=name,polygons=points.tolist(),row_threshold=row_threshold,column_threshold=column_threshold,
                logic_points=logic.tolist())
    if any(r0>r1 or c0>c1 or min(r0,c0)<0 for r0,r1,c0,c1 in logic.tolist()):
        result["expected_error"]="invalid_input"
    return result

def main():
    cases=[]
    for name,rects in [
        ("single",[(0,0,100,50)]),
        ("grid",[(0,0,100,50),(100,0,200,50),(0,50,100,100),(100,50,200,100)]),
        ("colspan",[(0,0,200,50),(0,50,100,100),(100,50,200,100)]),
        ("rowspan",[(0,0,100,100),(100,0,200,50),(100,50,200,100)]),
        ("missing-last-column",[(0,0,100,50),(100,0,200,50),(0,50,200,100),(200,50,300,100)]),
        ("column-insertion",[(0,0,100,50),(100,0,300,50),(0,50,150,100),(150,50,300,100)]),
        ("gap",[(0,0,90,50),(100,0,190,50),(100,50,190,100)]),
        ("adjacent-row-threshold",[(0,0,100,40),(100,10,200,50),(200,20,300,60)]),
        ("merge-threshold",[(0,0,200,90),(0,100,100,200),(100,100,200,200)]),
    ]:cases.append(capture(name,[polygon(*r)for r in rects]))
    rng=np.random.default_rng(176)
    for index in range(20):
        xs=np.cumsum([0,*rng.uniform(40,120,5)]);ys=np.cumsum([0,*rng.uniform(35,90,4)])
        polygons=[]
        for row in range(4):
            column=0
            while column<5:
                span=min(int(rng.integers(1,4)),5-column)
                polygons.append(polygon(xs[column],ys[row],xs[column+span],ys[row+1]));column+=span
        cases.append(capture(f"random-spans-{index}",polygons))
    fixture=json.loads((ROOT/"build/migration/unet-baseline.json").read_text())
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/fixture["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    pre=TSRUnetStructurer.__new__(TSRUnetStructurer);pre.inp_height=pre.inp_width=1024
    pre.mean=np.array([123.675,116.28,103.53],np.float32);pre.std=np.array([58.395,57.12,57.375],np.float32)
    for case in fixture["png_cases"]:
        image=np.asarray(Image.open(case["input"]).convert("RGB"))[:,:,::-1].copy()
        mask=session.run(None,{"input":pre.preprocess(image)["img"]})[0][0,0].astype(np.uint8)
        _,rotated=pre.postprocess(image,mask)
        if not rotated.size:continue
        polygons=rotated.reshape(-1,4,2).copy();polygons[:,[1,3]]=polygons[:,[3,1]]
        _,indices=sorted_ocr_boxes([box_4_2_poly_to_box_4_1(p)for p in polygons],threhold=.4)
        cases.append(capture("model-"+Image.open(case["input"]).filename,polygons[indices]))
    write_json(ROOT/"build/migration/table-recover.json",dict(cases=cases,scope="float64 polygon API; line-mask extraction and OCR text matching tested separately"))
    print(f"Exported {len(cases)} polygon topology cases")

if __name__=="__main__":main()
