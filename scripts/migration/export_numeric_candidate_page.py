"""Freeze combined structural page including partial-grid, year and metric replacements."""
import json
from common import ROOT,digest,write_json
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/native-numeric-candidate-page.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for name in ['native-numeric-candidate-segments','native-structural-table-expanded-page']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:cases.append(dict(label=name+'/'+c['label'],request=c['request'],expected=reference(c['request'],ordinary=True,supplemental=True,numeric=True)))
    refs=f['sources'].copy();p='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_open_grid.py';refs[p]=digest((ROOT/p).read_bytes());write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'combined facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
