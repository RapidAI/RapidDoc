"""Additional rotated and Unicode continuation-caption boundaries; preserve primary references."""
import copy,json
from common import ROOT,write_json
from export_native_table_page_policies import reference
from export_native_table_candidate_merge import annotation

def main():
    target=ROOT/'build/migration/native-table-page-policy-boundaries.json'
    if target.exists():raise FileExistsError(target)
    original=json.loads((ROOT/'build/migration/native-table-page-policies.json').read_text(encoding='utf-8'));base=next(c['request']for c in original['cases']if c['label']=='continuation-Table 1 continued');cases=[]
    def add(label,q):q=copy.deepcopy(q);cases.append(dict(label=label,request=q,expected=reference(q)))
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    for angle in [0,90,180,270]:
        for mode in ['new','existing','overwrite']:
            q=copy.deepcopy(base);s=q['source'];c=q['candidates'][0]
            def rotate(b):return list(_rotate_bbox_from_upright(tuple(b),tuple(s['page_size']),angle))
            c['angle']=angle;c['bbox']=rotate(c['bbox'])
            if mode!='new':c['annotations']=[annotation('note'),annotation()]
            if mode=='overwrite':s['lines'][0]['source_index']=8
            for a in c['annotations']:
                a['bbox']=rotate(a['bbox'])
                for pair in a['line_bboxes']:pair[1]=rotate(pair[1])
            for l in s['lines']:l['angle']=angle;l['bbox']=rotate(l['bbox'])
            for r in s['drawing_lines']:
                r['bbox']=rotate(r['bbox'])
                if angle in [90,270]:r['orientation']='vertical'if r['orientation']=='horizontal'else'horizontal'
            add('rotated-'+str(angle)+'-'+mode,q)
    for text in ['Table 1contİnued','Table 1contınued','Table İ continued','Table ı cont.','tab. XI continued','Table 1CC continued','Table 1cccontinued','Table 1 continued.','Table 1\ncontinued','\x1cTable 1 cont.\x1c','续\n表 １２','续表١','续表IV','续表ı','续表十一','Table １ continued']:
        q=copy.deepcopy(base);q['source']['lines'][0]['text']=text;add('unicode-'+repr(text),q)
    write_json(target,dict(sources=original['sources'],scope='Four rotations, first caption annotation/ordered overwrite and Python IGNORECASE/whitespace continuation boundaries.',cases=cases));print('Frozen policy boundaries',len(cases))
if __name__=='__main__':main()
