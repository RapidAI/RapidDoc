"""固定 TableMatch 几何合同及真实表格图像的 OCR→格归属。"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.utils.process_img import get_rotate_crop_image
from rapid_doc.model.table.rapid_table_self.table_matcher.main import TableMatch, _normalize_cell_bboxes
from common import ROOT, write_json

def expected(cells, boxes):
    cells = _normalize_cell_bboxes(cells)
    keep = [i for i,b in enumerate(boxes) if len(cells) and b[3] >= cells[:,1].min()]
    matched = TableMatch().match_result(cells,np.array([boxes[i] for i in keep]))
    result=[[] for _ in cells]
    for cell,indices in matched.items(): result[cell]=[keep[i] for i in indices]
    assigned={i for indices in result for i in indices}
    return dict(cells=result,unmatched=[i for i in range(len(boxes)) if i not in assigned])

def main():
    rng=np.random.default_rng(751);cases=[]
    for n in [0,1,2,17,257]:
        cells=[]
        for i in range(n):
            x,y=rng.uniform(-50,500,2);w,h=rng.uniform(0,80,2)
            cells.append([x,y,x+w,y+h] if i%2 else [x,y,x+w,y,x+w,y+h,x,y+h])
        boxes=[]
        for _ in range(300):
            x,y=rng.uniform(-100,600,2);w,h=rng.uniform(0,80,2);boxes.append([x,y,x+w,y+h])
        cases.append(dict(id=f"random-{n}",cell_bboxes=cells,ocr_boxes=boxes,expected=expected(cells,boxes)))
    for name,cells,boxes in [
        ("ties",[[0,0,10,10],[0,0,10,10],[10,0,20,10]],[[1,1,9,9],[10,0,10,9],[11,1,19,9],[0,-10,5,-1]]),
        ("threshold",[[0,0,10000,10000]],[[0,0,1,1],[0,0,1.001,1],[0,0,.999,1]]),
        ("zero",[[0,0,0,0]],[[0,0,0,0],[0,0,1,1]]),
    ]: cases.append(dict(id=name,cell_bboxes=cells,ocr_boxes=boxes,expected=expected(cells,boxes)))
    fixture=json.loads((ROOT/"build/migration/table-structure.json").read_text(encoding="utf-8"))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    runtime=lambda path:ort.InferenceSession(str(ROOT/path),sess_options=options,providers=["CPUExecutionProvider"])
    detector=runtime("rapid_doc/resources/PP-OCRv6_det_small.onnx");recognizer=runtime("rapid_doc/resources/PP-OCRv6_rec_small.onnx")
    decoder=CTCLabelDecode(character=recognizer.get_modelmeta().custom_metadata_map["character"].splitlines())
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    pngs={}
    for case in fixture["models"][0]["png_cases"]:
        image=np.asarray(Image.open(case["input"]).convert("RGB"))[:,:,::-1].copy()
        tensor=DetPreProcess(960,"max",[.485,.456,.406],[.229,.224,.225])(image)
        pred=detector.run(None,{detector.get_inputs()[0].name:tensor})[0]
        boxes,_=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)(pred,image.shape[:2])
        boxes=sorted(boxes,key=lambda b:b[0][1]);groups=[]
        for box in boxes:
            if not groups or abs(box[0][1]-groups[-1][-1][0][1])>=10:groups.append([])
            groups[-1].append(box)
        boxes=[b for group in groups for b in sorted(group,key=lambda b:b[0][0])]
        lines=[]
        for box in boxes:
            crop=get_rotate_crop_image(image,box.astype(np.float32))
            tensor=rec.resize_norm_img(crop,max(320/48,crop.shape[1]/crop.shape[0]))[None]
            text,score=decoder(recognizer.run(None,{recognizer.get_inputs()[0].name:tensor})[0])[0][0]
            lines.append(dict(points=box.tolist(),text=text,confidence=float(score)))
        pngs[case["input"]]=lines
    real=[]
    for model in fixture["models"]:
        for case in model["png_cases"]:
            lines=pngs[case["input"]]
            boxes=[[min(p[0] for p in line["points"]),min(p[1] for p in line["points"]),max(p[0] for p in line["points"]),max(p[1] for p in line["points"])]for line in lines]
            value=dict(variant=model["variant"],model=model["model"],sha256=model["sha256"],input=case["input"],structure=case,ocr_lines=lines,expected=expected(case["cell_bboxes"],boxes))
            if case["input"].endswith("blank.png"):
                # 固定当前两模型实际坏输出：plus 的 section 标签不配对，1m 的轴框反序。
                if model["variant"]=="slanet-plus":
                    stack=[];malformed=False
                    for token in case["tokens"]:
                        if token in ("<thead>","<tbody>"):stack.append(token)
                        elif token in ("</thead>","</tbody>"):
                            if not stack or stack.pop()!=token.replace("</","<"):malformed=True
                    assert malformed or stack
                else:assert any(b[0]>b[2] or b[1]>b[3] for b in case["cell_bboxes"])
                value["expected_error"]="invalid_input"
            real.append(value)
    write_json(ROOT/"build/migration/table-match.json",dict(cases=cases,png_cases=real))
    print(f"Exported {len(cases)} geometry contracts and {len(real)} real table OCR matching cases")

if __name__=="__main__":main()
