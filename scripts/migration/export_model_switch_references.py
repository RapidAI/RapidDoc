"""Capture Python model-adapter references for the native switching implementation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rapid_doc.model.table.docvortex_ocr import _deskew,extract_raster_rules,recover_ocr_table
from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer import structure
from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer.main import LABELS,THRESHOLDS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'tests/fixtures/model_switching');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    raster=[]
    base=np.full((300,440,3),255,np.uint8)
    for x in [20,120,260,420]:cv2.line(base,(x,20),(x,280),(0,0,0),2)
    for y in [20,80,160,220,280]:cv2.line(base,(20,y),(420,y),(0,0,0),2)
    chars=[dict(char=f'R{r}C{c}',bbox=[x+15,y+15,x+65,y+35],char_idx=r*3+c)for r,y in enumerate([20,80,160,220])for c,x in enumerate([20,120,260])]
    for angle in [0.,1.8,-2.4]:
        m=cv2.getRotationMatrix2D((220,150),angle,1.);image=cv2.warpAffine(base,m,(440,300),borderValue=(255,255,255))
        words=[]
        for word in chars:
            x0,y0,x1,y1=word['bbox'];p=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]])@m.T
            words.append(dict(word,bbox=[*p.min(axis=0),*p.max(axis=0)]))
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);edge=cv2.Canny(gray,50,150)
        corrected,transformed,skew=_deskew(image,words);rules=extract_raster_rules(corrected,transformed)
        filename=f'raster-{angle}.png';Image.fromarray(image).save(out/filename)
        boxes=[[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]for b in [w['bbox']for w in words]]
        ocr=[boxes,[w['char']for w in words],[.99]*len(words)]
        report={};result=recover_ocr_table(image,ocr,diagnostics=report)
        raster.append(dict(image=filename,words=[dict(bbox=w['bbox'],text=w['char'],source_index=i)for i,w in enumerate(words)],
            gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),canny_sha256=hashlib.sha256(edge.tobytes()).hexdigest(),
            angle=skew,width=corrected.shape[1],height=corrected.shape[0],rules=[dict(bbox=r.bbox,width=r.width,orientation=r.orientation)for r in rules],
            ocr=ocr,accepted=result is not None,html=result.html if result else None,diagnostics=report))
    import importlib.util
    from html import unescape
    spec=importlib.util.spec_from_file_location('ocr_rule_grid',ROOT/'tests/test_docvortex_ocr_table.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for kind in ['grid','merged','wireless','crossing']:
        image,ocr=module.grid(merged=kind=='merged',blank_row=kind=='merged',lines=kind!='wireless')
        if kind=='crossing':ocr[0][0]=[[20,20],[180,20],[180,36],[20,36]]
        words=[]
        for i,(box,value)in enumerate(zip(ocr[0],ocr[1])):
            b=np.asarray(box,dtype=float);words.append(dict(char=unescape(value),bbox=[*b.min(axis=0),*b.max(axis=0)],char_idx=i))
        corrected,transformed,skew=_deskew(image,words);rules=extract_raster_rules(corrected,transformed)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);edge=cv2.Canny(gray,50,150)
        filename=f'raster-{kind}.png';Image.fromarray(image).save(out/filename)
        report={};result=recover_ocr_table(image,ocr,diagnostics=report)
        raster.append(dict(image=filename,words=[dict(bbox=w['bbox'],text=w['char'],source_index=i)for i,w in enumerate(words)],
            gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),canny_sha256=hashlib.sha256(edge.tobytes()).hexdigest(),
            angle=skew,width=corrected.shape[1],height=corrected.shape[0],rules=[dict(bbox=r.bbox,width=r.width,orientation=r.orientation)for r in rules],
            ocr=ocr,accepted=result is not None,html=result.html if result else None,diagnostics=report))
    (out/'raster.json').write_text(json.dumps(raster,ensure_ascii=False,indent=2),encoding='utf-8')
    cases=[]
    rng=np.random.default_rng(90210)
    for seed in range(40):
        objs=[dict(label='table',score=.95,bbox=[0.,0.,400.,300.])]
        for label,axis,step,n in [('table row',1,60,5),('table column',0,100,4)]:
            for i in range(n):
                bbox=[0.,i*step,400.,(i+1)*step]if axis else[i*step,0.,(i+1)*step,300.]
                objs.append(dict(label=label,score=float(rng.uniform(.4,.99)),bbox=bbox))
                if seed%3==0:objs.append(dict(label=label,score=.3,bbox=bbox.copy()))
        objs.append(dict(label='table column header',score=.95,bbox=[0.,0.,400.,60.]))
        for _ in range(seed%5):
            r0=int(rng.integers(0,4));r1=int(rng.integers(r0+1,6));c0=int(rng.integers(0,3));c1=int(rng.integers(c0+1,5))
            objs.append(dict(label=str(rng.choice(['table spanning cell','table projected row header'])),score=float(rng.uniform(.4,.99)),bbox=[c0*100.,r0*60.,c1*100.,r1*60.]))
        tokens=[]if seed%7==0 else[dict(bbox=[c*100.+10.,r*60.+10.,c*100.+80.,r*60.+40.],text=f'{r},{c} & <x>',block_num=0,line_num=r*4+c,span_num=0)for r in range(5)for c in range(4)]
        structs=structure.objects_to_structures(deepcopy(objs),deepcopy(tokens),THRESHOLDS)
        cells=structure.structure_to_cells(structs[0],deepcopy(tokens))[0]if structs and structs[0]['rows']and structs[0]['columns']else[]
        cases.append(dict(objects=[dict(label=LABELS.index(o['label']),score=o['score'],bbox=o['bbox'])for o in objs],tokens=tokens,
            expected=[dict(bbox=c['bbox'],row_nums=c['row_nums'],column_nums=c['column_nums'],column_header=c['column header'],text=c['cell text'])for c in cells]))
    (out/'transformer.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    print('references',len(raster),'raster',len(cases),'TATR')

if __name__=='__main__':main()
