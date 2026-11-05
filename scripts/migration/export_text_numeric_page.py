"""Freeze joint page with all five newly ported numeric/text/partial branches."""
import json
from common import ROOT,digest,write_json
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/native-text-numeric-page.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for name in ['native-text-numeric-candidates','native-numeric-candidate-page']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:cases.append(dict(label=name+'/'+c['label'],request=c['request'],expected=reference(c['request'],ordinary=True,supplemental=True,numeric=True)))
    refs=f['sources'];write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'joint facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
