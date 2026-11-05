"""Original document title profiles, using the complete original page classifier."""
import copy,json
from dataclasses import asdict
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.document_profile import _infer_document_title_profile
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile,_PreparedPage
    b=q['body_profile'];b=None if b is None else _DocumentBodyProfile(**dict(b,regular_fonts=frozenset(tuple(f)for f in b['regular_fonts'])))
    pages=[_PreparedPage(page_size=tuple(p.get('page_size',p.get('source',{}).get('page_size'))),remaining_lines=lines(p['remaining_lines']),table_bboxes=[],drawing_lines=[],fixed_blocks=copy.deepcopy(p['fixed_blocks']))for p in q['pages']]
    p=_infer_document_title_profile(pages,b);return clean(None if p is None else asdict(p))

def main():
    target=ROOT/'build/migration/native-title-profiles.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-document-auxiliary-canonical.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if c['expected']['body_profile']is not None:add(c['label'],dict(pages=c['expected']['pages'],body_profile=c['expected']['body_profile']))
    b=dict(body_height=10.,body_weight=400.,regular_fonts=[['Body',0]],has_style_scale_repairs=False)
    def page():
        ls=[line('Original normal body paragraph with enough words',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,115.,130.,190.,205.,220.,280.,295.,310.,370.,385.,400.])]
        ls+=[line('Original section heading',[30.,y,180.,y+14.],20+i,h=14.,font='Title',weight=700.)for i,y in enumerate([160.,250.,340.])]
        ls.sort(key=lambda l:(l['bbox'][1],l['bbox'][0]));return dict(page_size=[400.,600.],remaining_lines=ls,fixed_blocks=[])
    for mode in ['positive','same-page','two-pages','one-seed','two-same-page','dedup-row','different-font','subset-font','different-flags','different-weight','unknown-weight','low-coverage','inside-container','inline-container','inline-bool','no-body','height-split','order','empty']:
        pages=[page()for _ in range(3)]
        for j,p in enumerate(pages):
            titles=[l for l in p['remaining_lines']if l['source_index']>=20]
            if mode in ['one-seed','two-same-page']:p['remaining_lines']=[l for l in p['remaining_lines']if l['source_index']<20 or l['source_index']<20+(1 if mode=='one-seed'else 2)]
            for l in titles:
                if mode=='dedup-row':l['visual_row_id']=10
                if mode=='different-font':l['font_signature']=[f'Title{j}',0]
                if mode=='subset-font':l['font_signature']=['ABCDEF+Title',0]
                if mode=='different-flags':l['font_signature']=['Title',j]
                if mode=='different-weight':l['dominant_font_weight']=[400.,600.,900.][j]
                if mode=='unknown-weight':l['dominant_font_weight']=None
                if mode=='low-coverage':l['font_coverage']=.5
                if mode=='height-split':l['effective_height']=l['em_height']=14.+j*3.
            if mode in ['inside-container','inline-container','inline-bool']:
                p['fixed_blocks']=[dict(type='image',bbox=[20.,140.,200.,180.])]
                if mode!='inside-container':p['fixed_blocks'][0]['_inline_visual_row_id']=True if mode=='inline-bool'else 10
            from export_native_late_image_text import project
            p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])]
        if mode in ['same-page','one-seed','two-same-page','dedup-row']:pages=pages[:1]
        if mode=='two-pages':
            pages=pages[:2]
            for p in pages:p['remaining_lines']=[l for l in p['remaining_lines']if l['source_index']<20 or l['source_index']==20]
        if mode=='empty':pages=[]
        if mode=='order':pages.reverse()
        add('directed/'+mode,dict(pages=pages,body_profile=None if mode=='no-body'else b))
    positive=next(c for c in cases if c['label']=='directed/positive');assert positive['expected']['prototypes'];assert positive['expected']['prototypes'][0]['support_pages']>=2
    assert next(c for c in cases if c['label']=='directed/one-seed')['expected']is None
    write_json(target,dict(cases=cases));print(len(cases),'complete original title-profile probes;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
