"""Original title-profile -> structural titles -> recurrent unknown-weight sequence."""
import copy,json
from dataclasses import asdict
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.document_profile import _infer_document_title_profile
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.structural import _classify_document_structural_titles,_classify_recurrent_unknown_weight_titles
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile,_PreparedPage
    def profile(b):return None if b is None else _DocumentBodyProfile(**dict(b,regular_fonts=frozenset(tuple(f)for f in b['regular_fonts'])))
    pages=[_PreparedPage(page_size=tuple(p.get('page_size',p.get('source',{}).get('page_size'))),remaining_lines=lines(p['remaining_lines']),table_bboxes=[],drawing_lines=[],fixed_blocks=copy.deepcopy(p['fixed_blocks']),canonical_formula_geometry=p.get('canonical_formula_geometry',False))for p in q['pages']]
    b=profile(q['body_profile']);canonical=profile(q['canonical_body_profile']);title=_infer_document_title_profile(pages,b)
    _classify_document_structural_titles(pages,canonical,legacy_body_profile=b,document_title_profile=title)
    _classify_recurrent_unknown_weight_titles(pages)
    raw=copy.deepcopy(q['pages'])
    for p,typed in zip(raw,pages):
        p['remaining_lines']=[project(l)for l in typed.remaining_lines]
        if 'canonical_formula_geometry'in p or typed.canonical_formula_geometry:p['canonical_formula_geometry']=typed.canonical_formula_geometry
    return clean(dict(pages=raw,title_profile=None if title is None else asdict(title)))

def main():
    target=ROOT/'build/migration/native-document-titles.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q)
        for p in q['pages']:p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])]
        c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    for c in json.loads((ROOT/'build/migration/native-document-auxiliary-canonical.json').read_text(encoding='utf-8'))['cases']:
        e=c['expected'];add(c['label'],dict(pages=e['pages'],body_profile=e['body_profile'],canonical_body_profile=e['canonical_body_profile']))
    for c in json.loads((ROOT/'build/migration/native-title-profiles.json').read_text(encoding='utf-8'))['cases']:
        if c['label'].startswith('directed/'):add(c['label'],dict(c['request'],canonical_body_profile=c['request']['body_profile']))
    b=dict(body_height=10.,body_weight=400.,regular_fonts=[['Body',0]],has_style_scale_repairs=True)
    def page():
        ls=[line('Original complete native body paragraph text',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,115.,130.,190.,205.,220.,280.,295.,310.,370.,385.,400.])]
        ls+=[line('Original independent heading',[30.,y,170.,y+20.],20+i,h=10.,font='Title',weight=700.)for i,y in enumerate([155.,245.,335.])]
        for l in ls:
            l['source_bbox']=l['bbox'][:];l['style_scale_repaired']=True
        ls.sort(key=lambda l:(l['bbox'][1],l['bbox'][0]));return dict(page_size=[400.,600.],remaining_lines=ls,fixed_blocks=[])
    for mode in ['canonical-positive','canonical-same-page','regular','no-repair','legacy-none','null-font','start-title','container','inline-container','low-coverage','source-bbox','rotated','font-rounding','near-anchor']:
        ps=[page()for _ in range(3)]
        for p in ps:
            for l in p['remaining_lines']:
                if l['source_index']<20:continue
                if mode=='regular':l['font_signature']=['Body',0]
                if mode=='null-font':l['font_signature']=None
                if mode=='start-title':l['bbox'][1]-=130.;l['bbox'][3]-=130.;l['source_bbox']=l['bbox'][:]
                if mode=='low-coverage':l['font_coverage']=.6
                if mode=='source-bbox':l['bbox'][3]=l['bbox'][1]+10.
                if mode=='font-rounding':l['em_height']=10.125
            if mode in ['container','inline-container']:
                p['fixed_blocks']=[dict(type='image',bbox=[20.,140.,200.,180.])]
                if mode=='inline-container':p['fixed_blocks'][0]['_inline_visual_row_id']=10
            if mode=='near-anchor':p['remaining_lines'].append(dict(line('Original nearby heading',[30.,177.,300.,197.],30,h=10.,font='Title',weight=700.),font_coverage=.6))
            if mode=='rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                p['page_size']=[600.,400.]
                for l in p['remaining_lines']:
                    l['angle']=90
                    for key in ['bbox','source_bbox']:
                        if key in l:l[key]=_rotate_bbox_from_upright(tuple(l[key]),tuple(p['page_size']),90)
        if mode=='canonical-same-page':ps=ps[:1]
        add(mode,dict(pages=ps,body_profile=None if mode=='legacy-none'else b,canonical_body_profile=dict(b,has_style_scale_repairs=mode!='no-repair')))
    for mode in ['unknown-positive','one-page','only-two','body-style','known-weight','terminal','long-group','font-change','container','outward','cjk-title']:
        ps=[]
        for index in range(3):
            ls=[]
            for group,y in enumerate([100.,250.,400.]):
                ls.append(line('Original heading words'if mode!='cjk-title'else'原始标题文本',[30.,y,170.,y+10.],group*10,font='Heading',weight=400. if mode=='known-weight'else None))
                for r in range(4):ls.append(line('Original paragraph contains enough English words for body',[30.,y+20.+r*15.,370.,y+30.+r*15.],group*10+r+1,font='Heading'if mode=='body-style'else'Body',weight=None))
                if mode=='terminal':ls[-5]['text']='Original heading words.'
                if mode=='font-change':ls[-5]['font_signature']=[f'Heading{index}',0]
                if mode=='outward':ls[-5]['bbox'][0]=200.;ls[-5]['bbox'][2]=370.
            if mode=='only-two':ls=[l for l in ls if l['source_index']<10]
            if mode=='long-group':
                for i in range(3):ls.append(line('Original additional heading',[30.,111.+i*2.,170.,121.+i*2.],40+i,font='Heading',weight=None))
            ps.append(dict(page_size=[400.,600.],remaining_lines=sorted(ls,key=lambda l:(l['bbox'][1],l['bbox'][0])),fixed_blocks=[dict(type='image',bbox=[20.,90.,180.,115.])]if mode=='container'else[]))
        if mode=='one-page':ps=ps[:2]
        add(mode,dict(pages=ps,body_profile=None,canonical_body_profile=None))
    positive=next(c for c in cases if c['label']=='unknown-positive');assert any(l['structural_title']for p in positive['expected']['pages']for l in p['remaining_lines'])
    assert any(l['structural_title']for c in cases if c['label']=='canonical-positive'for p in c['expected']['pages']for l in p['remaining_lines'])
    write_json(target,dict(cases=cases));print(len(cases),'complete original document-title sequences;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
