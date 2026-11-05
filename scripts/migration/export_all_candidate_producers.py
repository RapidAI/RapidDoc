"""Freeze all candidate producer branches in old page order, before final page policies."""
import json
from common import ROOT,digest,write_json
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/native-all-candidate-producers.json'
    if target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    for name in ['native-final-candidate-producers','native-final-candidate-specials','native-text-numeric-page']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            try:expected=reference(c['request'],ordinary=True,supplemental=True,numeric=True,final_producers=True)
            except KeyError as exc:errors.append(dict(label=name+'/'+c['label'],request=c['request'],error='invalid_input',old_key=exc.args[0]));continue
            cases.append(dict(label=name+'/'+c['label'],request=c['request'],expected=expected))
    p='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py';refs={p:digest((ROOT/p).read_bytes())};write_json(target,dict(sources=refs,cases=cases,errors=errors));print('Frozen',len(cases),'joint facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',len(errors),'old exceptions')
if __name__=='__main__':main()
