"""Correct typed projection counterparts; preserve initial frozen diagnostics unchanged."""
import copy,json
from common import ROOT,write_json
from export_native_page_containers import forms_reference,page_reference

def main():
    for name,reference in [('native-form-selection',forms_reference),('native-container-table-page',page_reference)]:
        target=ROOT/f'build/migration/{name}-canonical.json'
        if target.exists():
            if name=='native-form-selection':
                print('Retain existing canonical Form corpus');continue
            raise FileExistsError(target)
        original=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'));f=copy.deepcopy(original)
        for c in f['cases']:
            if name=='native-container-table-page':
                for l in c['request']['source']['lines']:l.setdefault('angle',0)
            c['expected']=reference(c['request'])
        f['scope']='Typed source counterpart: FormInfo.bbox restored to tuple; continuous table-stage lines carry their required angle. Initial frozen diagnostic corpus remains unchanged.'
        write_json(target,f);print(target.name,len(f['cases']))

if __name__=='__main__':main()
