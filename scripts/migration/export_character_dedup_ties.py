"""固定Python数值字典相等及signed-zero去重平局补充。"""
import copy,json
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/character-dedup-ties.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    from rapid_doc._vendor.docvortex_core.document.pdf.text.dedup import deduplicate_chars
    f=json.loads((ROOT/'build/migration/character-dedup.json').read_text(encoding='utf-8'))
    base=next(c['chars'][0]for c in f['cases']if c['label']=='bucket:63');cases=[]
    for visible in (None,True,False):
        for fontsize in (0.,10.):
            rows=[copy.deepcopy(base),copy.deepcopy(base)]
            rows[0]['char_idx']=0;rows[0]['source_indices']=(0,);rows[0]['text_object_id']=0;rows[0]['font']['size']=fontsize;rows[0]['writing_angle']=0.
            rows[1]['char_idx']=1;rows[1]['source_indices']=(1,);rows[1]['text_object_id']=1;rows[1]['font']['size']=-fontsize if fontsize==0. else fontsize;rows[1]['writing_angle']=-0.
            if visible is not None:rows[1]['text_is_visible']=visible
            for c in rows:c['bbox']=Bbox(c['bbox'])
            cases.append(dict(label=f'{visible}:{fontsize}',chars=encode(rows),expected=encode(deduplicate_chars(rows))))
    source='rapid_doc/_vendor/docvortex_core/document/pdf/text/dedup.py';write_json(target,dict(sources={source:digest((ROOT/source).read_bytes())},cases=cases));print('Frozen',len(cases),'font and angle equality ties')
if __name__=='__main__':main()
