"""Strong-graphic source-object mutations must reach the original source used for commit."""
import copy,json
from common import ROOT,write_json
from export_native_table_page_commit import reference
def main():
    target=ROOT/'build/migration/native-table-page-commit-aliases.json'
    if target.exists():raise FileExistsError(target)
    old=json.loads((ROOT/'build/migration/native-strong-graphics-complete.json').read_text(encoding='utf-8'));base=next(c['request']for c in old['cases']if c['request']['source']!=c['expected']['source']);cases=[]
    for mode in ['caption-aliases','protected-caption','prior-claimed-caption']:
        q=copy.deepcopy(base)
        for i,l in enumerate(q['source']['lines']):l['source_index']=i;l.setdefault('angle',0);l.setdefault('effective_height',l['bbox'][3]-l['bbox'][1])
        if mode=='protected-caption':q['source']['lines'][-2]['semantic_type']='caption'
        if mode=='prior-claimed-caption':q['prior_claimed_line_indices']=[len(q['source']['lines'])-2]
        out=reference(q);cases.append(dict(label=mode,request=q,expected=out))
    assert cases[0]['request']['source']['lines']!=cases[0]['expected']['source']['lines']
    refs=json.loads((ROOT/'build/migration/native-table-page-commit.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,scope='Unique source indices and typed angles; actual old strong-caption alias side effects across filtered analysis and original commit source.',cases=cases,errors=[]));print('Frozen alias stage facts',len(cases))
if __name__=='__main__':main()
