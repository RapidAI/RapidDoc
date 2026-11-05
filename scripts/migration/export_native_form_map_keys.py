"""Exact Python dictionary key retention and canonical frozenset projection."""
import copy,json
from common import ROOT,write_json
from export_native_page_containers import form_base,forms_reference

def main():
    target=ROOT/'build/migration/native-form-map-keys.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for mode in ['integer-key','integer-alias','duplicate-numeric-key','unsorted-members','same-alias','negative-ids']:
        s=form_base();key=[40,80,360,360];s['form_bboxes']=[key];s['form_member_sources']=[dict(bbox=key,source_indices=[1,0])];s['form_path_sources']=[dict(bbox=key,source_indices=[1,0])]
        if mode=='integer-key':s['path_infos']=[]
        if mode=='integer-alias':s['form_member_sources'].append(dict(bbox=[60,100,340,340],source_indices=[99]))
        if mode=='duplicate-numeric-key':s['form_member_sources'].append(dict(bbox=[40.,80.,360.,360.],source_indices=[0,1]))
        if mode=='same-alias':s['form_bboxes']*=3
        if mode=='negative-ids':
            for l in s['lines']:
                for c in l['chars']:c['char_idx']-=100
            s['form_member_sources'][0]['source_indices']=[-100,-99]
        q=dict(source=s);cases.append(dict(label=mode,request=copy.deepcopy(q),expected=forms_reference(q)))
    refs=json.loads((ROOT/'build/migration/native-index-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases,errors=[]));print(len(cases),'old dictionary key/type facts')

if __name__=='__main__':main()
