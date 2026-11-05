"""Original same-baseline -> overlapping clusters -> same-baseline closure."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_image_chain import line
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_merging import merge_text_line_clusters
    return [project(l)for l in merge_text_line_clusters(lines(q['lines']),tuple(q['page_size']),[tuple(b)for b in q['table_bboxes']])]
def scenes():
    cases=[]
    def add(label,ls,**kw):
        q=dict(lines=[project(l)for l in lines(ls)],page_size=[400.,600.],table_bboxes=[]);q.update(kw);cases.append(dict(label=label,request=q,expected=reference(q)))
    def base():return [dict(line('Original native',[30.,100.,110.,110.],0),font_signature=['Body',0],font_coverage=1.),dict(line('body words',[115.,100.,180.,110.],1),font_signature=['Body',0],font_coverage=1.)]
    for mode in ['basic','different-font','low-coverage','semantic','formula-only','split-row','source-row','source-gap','preserve','caption-left','caption-right','bullet','same-band','different-band','paragraph-group','suppressed','unicode-label','empty-text','font-metrics','rotated']:
        ls=base();kw={}
        if mode=='different-font':ls[1]['font_signature']=['Other',0]
        if mode=='low-coverage':ls[1]['font_coverage']=.5
        if mode=='semantic':ls[1]['semantic_type']='text'
        if mode=='formula-only':ls[1]['formula_candidate_only']=True
        if mode=='split-row':
            for l in ls:l.update(visual_row_id=0,split_from_row=True)
        if mode in ['source-row','source-gap','preserve']:
            for i,l in enumerate(ls):l.update(baseline=110.,source_bbox=l['bbox'],chars=[dict(char='x',char_idx=i*(1 if mode!='source-gap'else 10),bbox=l['bbox'],font=dict(name='Body',size=10.,flags=0))])
            ls[1]['font_signature']=['Other',0]
            if mode=='preserve':ls[1]['preserve_split_boundary']=True
        if mode=='caption-left':ls[1]['caption_start']=True
        if mode=='caption-right':ls[0]['caption_start']=True
        if mode=='bullet':ls[0].update(text='•',bbox=[30.,100.,35.,110.]);ls[1]['bbox'][0]=40.
        if mode in ['same-band','different-band']:
            for i,l in enumerate(ls):l.update(title_band_id=4 if mode=='same-band'else i,structural_title=True,explicit_section_title=True)
        if mode=='paragraph-group':
            for l in ls:l['paragraph_group']=9
        if mode=='suppressed':
            for l in ls:l['title_suppressed']=True
        if mode=='unicode-label':ls[0]['text']='٣.';ls[1]['native_title_label_left']=117.
        if mode=='empty-text':ls[1]['text']=' '
        if mode=='font-metrics':
            for l in ls:l['chars']=[dict(char=c,char_idx=i+l['source_index']*100,bbox=[l['bbox'][0]+i*5.,100.,l['bbox'][0]+i*5.+5.,110.],font=dict(name='Body',size=10.,flags=0,weight=400.))for i,c in enumerate(l['text'])]
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            kw['page_size']=[600.,400.]
            for l in ls:l['bbox']=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(kw['page_size']),90));l['angle']=90
        add(mode,ls,**kw)
    for dx in [-2.6,-2.5,-1.5,-.01,0.,.5,.75,.75001,3.,7.5,7.50001]:
        for dy in [0.,2.5,3.,3.00001]:
            ls=base();ls[1]['bbox']=[110.+dx,100.+dy,180.+dx,110.+dy];add(f'geometry/{dx}/{dy}',ls)
    for b in [[105.,95.,120.,115.],[0.,0.,10.,10.]]:add('table/'+str(b),base(),table_bboxes=[b])
    f=json.loads((ROOT/'build/migration/native-inline-clusters.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'lines'in c['request']and 'page_size'in c['request']:add('cluster/'+c['label'],c['request']['lines'],page_size=c['request']['page_size'],table_bboxes=c['request'].get('table_bboxes',[]))
    return cases
def main():
    p=ROOT/'build/migration/native-page-merging.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=scenes();write_json(p,dict(cases=cases));print(len(cases),'old page merge closures')
if __name__=='__main__':main()
