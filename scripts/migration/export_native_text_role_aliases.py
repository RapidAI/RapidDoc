"""Object alias, stable tie and canonical-source evidence for new role classifiers."""
import copy,json
from common import ROOT,write_json
from export_native_text_roles import scenes,reference
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_geometry_pages import reference as finish
from export_native_table_page_policies import empty

def main():
    target=ROOT/'build/migration/native-text-role-aliases-canonical.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[];pages=[]
    def add(label,q):cases.append(dict(label=label,request=copy.deepcopy(q),expected=reference(q)))
    originals=scenes();quote=next(c['request']for c in originals if c['label']=='quote/italic')
    ls=[]
    for k in range(3):
        intro=copy.deepcopy(quote['lines'][0]);intro.update(text='Original source explains:',bbox=[30.,60.+k*120,230.,70.+k*120],source_index=100+k,font_signature=['Label',0],dominant_font_weight=500.,visual_row_id=None);ls.append(intro)
        for j,l in enumerate(quote['lines']):
            l=copy.deepcopy(l);l.update(bbox=[30.,80.+k*120+j*16,330.,90.+k*120+j*16],source_index=k*10+j,visual_row_id=k*10+j);ls.append(l)
    q=copy.deepcopy(quote);q.update(lines=ls,reference_lines=copy.deepcopy(ls),reference_aliases=[dict(reference_index=i,line_index=i)for i in range(len(ls))]);add('regular-reference-aliases',q);q['reference_aliases']=[];add('regular-reference-detached',q)
    assert cases[-1]['expected']!=cases[-2]['expected']
    q=copy.deepcopy(next(c['request']for c in originals if c['label']=='bilingual/positive'));prior=copy.deepcopy(q['lines'][2]);prior.update(text='Earlier right body',bbox=[100.,80.,280.,90.],source_index=20);left=copy.deepcopy(prior);left.update(text='Earlier left body',bbox=[30.,80.,280.,90.],source_index=21);q['lines']=[prior,left]+q['lines'];q['reference_lines']=copy.deepcopy(q['lines']);q['reference_aliases']=[dict(reference_index=i,line_index=i)for i in range(len(q['lines']))];add('stable-previous-tie',q)
    def page(label,q):
        q=copy.deepcopy(q);q['include_post_text_roles']=True;pages.append(dict(label=label,request=q,expected=finish(q,post_text=True)))
    f=json.loads((ROOT/'build/migration/native-geometry-aliases.json').read_text(encoding='utf-8'))
    for c in f['cases']:page(c['label'],c['request'])
    for anomaly in [False,True]:
        ls=copy.deepcopy(quote['lines']);p=dict(source=dict(empty(),lines=ls,page_size=quote['page_size']),remaining_lines=copy.deepcopy(ls),table_bboxes=[],strong_graphic_bboxes=[],claimed_line_indices=[])
        page('quotation-canonical/'+str(anomaly),dict(prepared=p,page_index=0,plan=dict(document_style_anomaly=anomaly),style_lines=[],link_lines=[]))
        assert len(pages[-1]['expected']['canonical_formula_source_lines'])==(3 if anomaly else 0)
    # Real visual-run input and exact plan, then classifications after detached runs.
    f=json.loads((ROOT/'build/migration/native-geometry-pages.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if c['label']in ['resplit/split','resplit/labels-into-images']:page(c['label'],c['request'])
    write_json(target,dict(cases=cases,pages=pages));print(len(cases),'alias/tie contracts;',len(pages),'geometry-to-role alias/canonical continuations')
if __name__=='__main__':main()
