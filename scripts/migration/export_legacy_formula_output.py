"""冻结启用旧公式编号后处理后的完整 Markdown/content_list 输出。"""
import copy,importlib,json
from common import ROOT,digest,write_json


def main():
    target=ROOT/'build/migration/legacy-formula-output.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    assert (old.inline_left_delimiter,old.inline_right_delimiter,old.display_left_delimiter,old.display_right_delimiter)==('$','$','$$','$$')
    source=json.loads((ROOT/'build/migration/legacy-formula-numbers.json').read_text());cases=[];excluded=[]
    def algorithm(v):
        if isinstance(v,dict):return v.get('original_label')=='algorithm'or any(algorithm(x)for x in v.values())
        if isinstance(v,list):return any(algorithm(x)for x in v)
        return False
    for c in source['chains']:
        if algorithm(c['expected']):excluded.append(dict(name=c['name'],reason='Magika 语言分类尚未迁移'));continue
        for mode in ['mm_markdown','nlp_markdown','content_list']:
            expected=old.union_make(copy.deepcopy(c['expected']),mode,'images')
            cases.append(dict(name=c['name']+'/'+mode,requests=c['requests'],make_mode=mode,expected=expected))
    paths=['rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,excluded=excluded,sources=source['sources']|{p:digest((ROOT/p).read_bytes())for p in paths},
        profile='旧模型页面→公式编号→para_split→union_make；默认公式分隔符、无资产 writer，算法保守排除'))
    print(f'Frozen {len(cases)} old model→formula→paragraph→output references; {len(excluded)} algorithm chains excluded')


if __name__=='__main__':main()
