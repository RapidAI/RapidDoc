"""实际旧 OpenCV/NumPy 对比度，覆盖 FP64 pairwise 块边界与阈值附近。"""
import importlib
import numpy as np
from common import ROOT,digest,write_json

if __name__=='__main__':
    target=ROOT/'build/migration/native-contrast.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    from rapid_doc.utils.span_pre_proc import calculate_contrast
    rng=np.random.default_rng(20261024);cases=[]
    for n in [1,2,7,8,9,127,128,129,257,1025]:
        for _ in range(4):
            rgb=rng.integers(0,256,(1,n,3),dtype=np.uint8);cases.append(dict(rgb=rgb.ravel().tolist(),expected=float(calculate_contrast(rgb,'rgb'))))
    for base in [0,1,10,100,200]:
        for delta in range(51):
            rgb=np.array([base,min(255,base+delta)]*64,dtype=np.uint8).reshape(1,128,1).repeat(3,axis=2)
            cases.append(dict(rgb=rgb.ravel().tolist(),expected=float(calculate_contrast(rgb,'rgb'))))
    p='rapid_doc/utils/span_pre_proc.py';write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())},numpy=np.__version__))
    print(f'Frozen {len(cases)} actual old contrast values')
