"""Retain original compact-footnote font conversion exceptions."""
import copy,json
from common import ROOT,write_json
from export_native_geometry_pages import reference
def main():
    p=ROOT/'build/migration/native-prepared-font-contracts.json'
    if p.exists():raise FileExistsError(p)
    f=json.loads((ROOT/'build/migration/native-prepared-identity.json').read_text(encoding='utf-8'));base=next(c['request']for c in f['cases']if c['label']=='compact/footnote-superscript/canonical');cases=[]
    for label,pos,field,value in [('marker-null-size',0,'size',None),('body-null-size',2,'size',None),('marker-null-font',0,None,None)]:
        q=copy.deepcopy(base);chars=q['prepared']['source']['lines'][0]['chars']
        if field:chars[pos]['font'][field]=value
        else:chars[pos]['font']=value
        q['prepared']['remaining_lines']=copy.deepcopy(q['prepared']['source']['lines']);c=dict(label=label,request=q)
        try:c['expected']=reference(q,post_text=True,full=True)
        except (TypeError,ValueError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    assert all('error'in c for c in cases)
    write_json(p,dict(cases=cases));print('3 original compact font exceptions')
if __name__=='__main__':main()
