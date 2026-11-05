"""固定空表、无 tbody 与含空白表头等跨页状态边界。"""
import copy,importlib,json
from common import ROOT,digest,write_json

if __name__=='__main__':
    target=ROOT/'build/migration/legacy-table-merge-edges.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    old=importlib.import_module('rapid_doc.utils.table_merge');cases=[]
    template=json.loads((ROOT/'build/migration/legacy-table-merge.json').read_text())['cases'][0]['pages']
    choices=['<table></table>','<div>no rows</div>','<table><tr></tr></table>','<table><tr><td></td></tr></table>','<table><tr><td>1</td></tr></table>']
    for a in choices:
        for b in choices:
            pages=copy.deepcopy(template)
            for p,html in zip(pages,[a,b]):p['para_blocks'][0]['blocks'][0]['lines'][0]['spans'][0]['html']=html
            expected=copy.deepcopy(pages);old.merge_table(expected);cases.append(dict(pages=pages,expected=expected))
    paths=['rapid_doc/utils/table_merge.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in paths}))
    print(f'Frozen {len(cases)} empty/boundary table states')
