"""Genuine three-column table rails next to code, retaining first rail-free old outcomes."""
import copy,json
from common import ROOT,write_json
from export_native_table_page_commit import reference
def main():
    target=ROOT/'build/migration/native-code-table-grid-mixtures.json'
    if target.exists():raise FileExistsError(target)
    old=json.loads((ROOT/'build/migration/native-code-table-page.json').read_text(encoding='utf-8'));cases=[]
    for c in old['cases']:
        if not c['label'].startswith('code-and-table/'):continue
        q=copy.deepcopy(c['request']);s=q['source']
        for l in s['lines']:
            if l['source_index']>=1000:l['visual_row_id']=round((l['bbox'][1]+l['bbox'][3])/2.)
        s['drawing_lines'] += [dict(bbox=[x,309.9,x,410.1],orientation='vertical',width=0.)for x in [40.,160.,250.,310.,380.]]
        out=reference(q,True);assert out['table_blocks'],c['label'];assert out['code_blocks']or out['rule_code_blocks'],c['label'];cases.append(dict(label=c['label']+'/physical-grid',request=q,expected=out))
    write_json(target,dict(sources=old['sources'],scope='Actual code plus numeric table source with physical vertical rails and true shared row identities. Rail-free earlier old misclassification outcomes remain immutable.',cases=cases,errors=[]));print('Frozen code/table physical mixtures',len(cases))
if __name__=='__main__':main()
