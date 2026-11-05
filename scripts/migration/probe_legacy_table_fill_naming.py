"""只读探测旧表内 UUID 图片哈希种子的进程地址差异，不修改旧实现。"""
import argparse,importlib,json,subprocess,sys
from common import ROOT,digest,write_json

def worker():
    from PIL import Image
    module=importlib.import_module('rapid_doc.utils.pdf_image_tools')
    inputs=[];original=module.str_sha256
    def capture(value):inputs.append(value);return original(value)
    class Writer:
        _parent_dir='images'
        def __init__(self):self.calls=[]
        def write(self,name,data):self.calls.append(dict(name=name,bytes_sha256=digest(data)))
    dets=[dict(category_id=5,html='<table><td>test-image-uuid</td></table>')];writer=Writer()
    module.str_sha256=capture
    try:
        with Image.new('RGB',(2,2),(10,20,30))as image:
            module.save_table_fill_image(dets,[dict(uuid='test-image-uuid',bbox=[1,2,5,6],pil_image=image)],'FIXED_PAGE_MD5',7,writer)
    finally:module.str_sha256=original
    assert len(inputs)==len(writer.calls)==1
    print(json.dumps(dict(hash_inputs=inputs,calls=writer.calls,dets=dets)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:worker();return
    target=ROOT/'build/migration/legacy-table-fill-naming.json'
    if target.exists():raise FileExistsError('Frozen probe exists')
    cases=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker'],cwd=ROOT,text=True,encoding='utf-8'))for _ in range(3)]
    assert len({c['calls'][0]['bytes_sha256']for c in cases})==1
    distinct=len({c['calls'][0]['name']for c in cases})
    source='rapid_doc/utils/pdf_image_tools.py'
    write_json(target,dict(cases=cases,distinct_names=distinct,source=source,source_sha256=digest((ROOT/source).read_bytes()),
        scope='实际旧函数；仅包装原 SHA256 记录入参并转调原函数；固定图片/页号/bbox/页面 hash；不同子进程；不归一化函数地址'))
    print(f'Old table-fill naming: {distinct} names across 3 processes for identical PNG bytes; captured seeds in {target}')

if __name__=='__main__':main()
