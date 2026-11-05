"""旧 RGB 页图 ICC/EXIF/透明色不改变 NumPy 补识别裁图像素。"""
import copy,importlib,json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

if __name__=='__main__':
    target=ROOT/'build/migration/legacy-native-text-metadata.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');old=importlib.import_module('rapid_doc.utils.span_pre_proc')
    base=json.loads((ROOT/'build/migration/legacy-native-text.json').read_text());request=next(c['request']for c in base['cases']if c['name'].startswith('contrast/'))
    paths=[]
    for c in json.loads((ROOT/'build/migration/legacy-assets-metadata.json').read_text())['cases']:
        with Image.open(ROOT/c['source'])as image:
            if image.mode=='RGB'and c['source']not in paths:paths.append(c['source'])
    cases=[]
    for path in paths:
        r=copy.deepcopy(request);page=copy.deepcopy(r['native_page'])
        for b in page['blocks']:
            for line in b['lines']:line['bbox']=SimpleNamespace(bbox=line['bbox'])
        with Image.open(ROOT/path)as image:
            expected=old.txt_spans_extract(page,copy.deepcopy(r['spans']),image,r['scale'],r['blocks'],r['discarded_blocks'])
        crops=[]
        for s in expected:
            if 'np_img'in s:
                a=s['np_img'];name='ocr-7-'+str(len(crops));crops.append(dict(name=name,width=a.shape[1],height=a.shape[0],rgb_sha256=digest(a[:,:,::-1].copy().tobytes())));s['np_img']=name
        cases.append(dict(name=path,source=path,request=r,expected=expected,crops=crops,assets=[]))
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/pdf_image_tools.py']
    write_json(target,dict(cases=cases,pages=[],sources={p:digest((ROOT/p).read_bytes())for p in refs},inputs={p:digest((ROOT/p).read_bytes())for p in paths}))
    print(f'Frozen {len(cases)} old native OCR crop metadata references')
