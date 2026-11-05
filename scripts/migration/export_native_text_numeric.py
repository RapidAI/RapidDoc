"""旧稳定 top 排序的正负零边界，避免 total_cmp 改变字符归属。"""
import copy,importlib,json
from types import SimpleNamespace
from PIL import Image
from common import ROOT,digest,write_json

if __name__=='__main__':
    target=ROOT/'build/migration/legacy-native-text-numeric.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');old=importlib.import_module('rapid_doc.utils.span_pre_proc')
    base=json.loads((ROOT/'build/migration/legacy-native-text-narrow.json').read_text())['cases'][0];cases=[]
    for values in [[0.,-0.],[-0.,0.],[0,-0.]]:
        r=copy.deepcopy(base['request']);s=copy.deepcopy(r['spans'][0]);s['bbox']=[10,values[0],30,10];s['marker']='first'
        t=copy.deepcopy(s);t['bbox'][1]=values[1];t['marker']='second';r['spans']=[s,t]
        c=dict(char='A',char_idx=0,bbox=[12,1,18,9]);r['native_page']=dict(blocks=[dict(lines=[dict(rotation=0,bbox=[0,0,100,100],spans=[dict(text='A',chars=[c])])])])
        page=copy.deepcopy(r['native_page']);page['blocks'][0]['lines'][0]['bbox']=SimpleNamespace(bbox=[0,0,100,100])
        with Image.open(ROOT/base['source'])as image:expected=old.txt_spans_extract(page,copy.deepcopy(r['spans']),image,1.,r['blocks'],r['discarded_blocks'])
        crops=[]
        for span in expected:
            if 'np_img'in span:
                a=span['np_img'];name='ocr-7-'+str(len(crops));crops.append(dict(name=name,width=a.shape[1],height=a.shape[0],rgb_sha256=digest(a[:,:,::-1].copy().tobytes())));span['np_img']=name
        cases.append(dict(name=repr(values),request=r,source=base['source'],expected=expected,crops=crops,assets=[]))
    p='rapid_doc/utils/span_pre_proc.py';write_json(target,dict(cases=cases,pages=[],inputs={base['source']:digest((ROOT/base['source']).read_bytes())},sources={p:digest((ROOT/p).read_bytes())}))
    print('Frozen 3 old signed-zero native character ownership cases')
