"""冻结旧单页输出→para_split 的单/多页串联；不覆盖已冻结参考。"""
import copy,importlib,json
from common import ROOT,write_json,digest


def main():
    target=ROOT/'build/migration/legacy-page-chain.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    fixture=json.loads((ROOT/'build/migration/legacy-page.json').read_text())
    split=importlib.import_module('rapid_doc.backend.pipeline.para_split').para_split
    cases=[]
    def add(name,items):
        requests=[copy.deepcopy(c['request'])for c in items];pages=[]
        for i,c in enumerate(items):
            requests[i]['page_index']=i*3
            page=copy.deepcopy(c['expected'])
            if page is None:page=dict(preproc_blocks=[],page_size=[int(requests[i]['page_width']),int(requests[i]['page_height'])],discarded_blocks=[])
            page['page_idx']=i*3;pages.append(page)
        split(pages);cases.append(dict(name=name,requests=requests,expected=pages))
    add('empty',[])
    for c in fixture['cases']:add('single/'+c['name'],[c])
    for i in range(0,len(fixture['cases'])-2,3):add(f'multi-{i}',fixture['cases'][i:i+3])
    paths=['rapid_doc/backend/pipeline/para_split.py']
    write_json(target,dict(cases=cases,sources=fixture['sources']|{p:digest((ROOT/p).read_bytes())for p in paths},
        profile='实际旧单页输出→para_split；空页按 result_to_middle_json 包装；无后置 OCR/公式编号/跨页表格合并'))
    print(f'Frozen {len(cases)} old page→paragraph chains')


if __name__=='__main__':main()
