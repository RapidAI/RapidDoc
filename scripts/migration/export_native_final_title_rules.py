"""Existing Python final-page title refinements, with positive evidence for each pass."""
import copy,json,sys
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line
from export_native_page_structure import BODY
KINDS=['front','suppress','runin','repeated','panels','numbered','wrapped','displaced','cjk']
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_merging as merge
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis import structural,panels,native_boundaries
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import group_front_matter_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile
    ls=lines(q['lines'])
    if '--canonical'in sys.argv:
        for l in ls:
            if l.wrapped_title_parts is not None:l.wrapped_title_parts=tuple(lines(list(l.wrapped_title_parts)))
    p=tuple(q['page_size']);images=[tuple(b)for b in q['image_bboxes']];kind=q['kind'];body=q.get('document_body_profile');body=None if body is None else _DocumentBodyProfile(**dict(body,regular_fonts=frozenset(tuple(f)for f in body['regular_fonts'])))
    if kind=='front':group_front_matter_lines(ls,p)
    elif kind=='suppress':native_boundaries.suppress_continuous_sentence_titles(ls)
    elif kind=='runin':merge._demote_runin_title_fragments(ls,p)
    elif kind=='repeated':structural._demote_regular_repeated_item_titles(ls,body)
    elif kind=='panels':panels.classify_panel_titles(ls,images,p[0])
    elif kind=='numbered':structural._classify_bold_numbered_heading_rows(ls,p,[tuple(b)for b in q['container_bboxes']])
    elif kind=='wrapped':ls=structural._restore_wrapped_bold_title_tails(ls,p)
    elif kind=='displaced':structural._restore_short_heading_with_image_displaced_prose(ls,images)
    elif kind=='cjk':structural._classify_short_cjk_section_leads(ls)
    else:raise ValueError(kind)
    return clean(dict(lines=[project(l)for l in ls]))
def main():
    typed='--typed'in sys.argv
    if typed:sys.argv.append('--canonical')
    if '--canonical'in sys.argv:
        target=ROOT/('build/migration/native-final-title-rules-typed.json'if typed else'build/migration/native-final-title-rules-canonical.json')
        if target.exists():raise FileExistsError(target)
        from loguru import logger
        logger.remove();f=json.loads((ROOT/'build/migration/native-final-title-rules.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if typed:
                for l in c['request']['lines']:
                    if l.get('wrapped_title_parts')is not None:l['wrapped_title_parts']=[project(p)for p in lines(l['wrapped_title_parts'])]
            for key in ['error','old_exception','expected']:c.pop(key,None)
            try:c['expected']=reference(c['request'])
            except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        write_json(target,f);print(len(f['cases']),'canonical typed wrapped-title cases;',sum('error'in c for c in f['cases']),'exceptions');return
    target=ROOT/'build/migration/native-final-title-rules.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('image_bboxes',[]);q.setdefault('container_bboxes',[]);q.setdefault('document_body_profile',BODY);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-page-structure.json').read_text(encoding='utf-8'))
    for i,c in enumerate(f['cases']):
        if i%17 or 'expected'not in c:continue
        for kind in KINDS:add('structure/'+c['label']+'/'+kind,dict(c['request'],lines=c['expected']['lines'],kind=kind))
    def base(kind):
        ls=[line('Original ordinary complete body words',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])];q=dict(lines=ls,kind=kind)
        if kind=='front':q['lines']=[line('1 Original University',[30.,120.,220.,130.],0),line('Original department and contact details',[40.,134.,300.,144.],1),line('Abstract',[30.,220.,120.,230.],2)]
        if kind=='suppress':q['lines']=[line('中文正文连续，说明条件仍然保持',[40.,180.,300.,190.],0,semantic_type='paragraph_title'),line('继续中文正文说明',[30.,194.,180.,204.],1)]
        if kind=='runin':q['lines'] += [line('Original lead',[30.,190.,100.,200.],20,font='Bold',weight=700.,semantic_type='paragraph_title'),line('ordinary continuation words',[105.,190.,300.,200.],21)]
        if kind=='repeated':q['lines'] += [line('1. Original first item',[30.,180.,180.,190.],20,semantic_type='paragraph_title'),line('2. Original second item',[30.,210.,180.,220.],21)]
        if kind=='panels':q['lines'] += [line('Left panel',[30.,190.,130.,202.],20,h=12.,font='Heading'),line('Right panel',[230.,190.,330.,202.],21,h=12.,font='Heading'),line('Original smaller panel body',[30.,210.,160.,220.],22),line('Original smaller panel body',[230.,210.,360.,220.],23)]
        if kind=='numbered':q['lines'] += [line('2 Original section heading',[30.,190.,260.,200.],20,font='Bold',weight=700.)]
        if kind=='wrapped':
            q['lines'] += [line('Original heading',[30.,175.,200.,185.],20,font='Bold',weight=700.,semantic_type='paragraph_title'),line('Tail. ordinary body words continue now',[30.,190.,300.,200.],21)]
            q['lines'][-1]['wrapped_title_parts']=[line('Tail.',[30.,190.,65.,200.],21,font='Bold',weight=700.),line('ordinary body words continue now',[70.,190.,300.,200.],21)]
        if kind=='displaced':q['lines']=[line('Original independent title',[30.,70.,200.,80.],0),line('Original complete body paragraph with at least eight ordinary words',[30.,100.,370.,110.],1),line('Original complete body paragraph with at least eight ordinary words',[30.,114.,370.,124.],2),line('ending.',[30.,220.,65.,230.],3)];q['image_bboxes']=[[70.,124.,370.,220.]]
        if kind=='cjk':q['lines']=[line('章节标题说明',[30.,180.,100.,190.],0),line('这是原生中文正文完整内容用于提供章节标题判断依据',[30.,200.,370.,210.],1),line('继续中文正文完整内容保证字体字号段落证据',[30.,214.,370.,224.],2)]
        return q
    for kind in KINDS:
        for mode in ['positive','empty','container','image','no-profile','semantic','suppressed','font','weight','margin','gap','unicode','rotated','null-font','duplicate','same-row','tiny','wrapped-chars']:
            q=base(kind)
            if mode=='empty':q['lines']=[]
            if mode=='container':q['container_bboxes']=[[20.,150.,380.,250.]]
            if mode=='image':q['image_bboxes']=[[20.,205.,180.,280.],[220.,205.,380.,280.]]
            if mode=='no-profile':q['document_body_profile']=None
            if mode in ['semantic','suppressed','font','weight','null-font','same-row','tiny']:
                for l in q['lines']:
                    if mode=='semantic':l['semantic_type']='caption'
                    if mode=='suppressed':l['title_suppressed']=True
                    if mode=='font':l['font_signature']=['Other',0]
                    if mode=='weight':l['dominant_font_weight']=400.
                    if mode=='null-font':l['font_signature']=None
                    if mode=='same-row':l['visual_row_id']=9
                    if mode=='tiny':l.update(effective_height=1.,em_height=1.)
            if mode in ['margin','gap']:
                for l in q['lines']:l['bbox'][1]+= -170. if mode=='margin'else 30.;l['bbox'][3]+= -170. if mode=='margin'else 30.
            if mode=='unicode':
                for l in q['lines']:l['text']=l['text'].replace('1.','١.').replace('2 ','２ ').replace(' ','\x1c')
            if mode=='duplicate':q['lines'].append(copy.deepcopy(q['lines'][-1]))
            if mode=='rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                q['page_size']=[600.,400.]
                for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
            if mode=='wrapped-chars'and kind=='wrapped':
                l=q['lines'][-1];l.pop('wrapped_title_parts',None);l['chars']=[dict(char=c,char_idx=i,bbox=[30.+i*4,190.,34.+i*4,200.],font=dict(name='Bold'if i<5 else'Body',size=10.,flags=0,weight=700. if i<5 else 400.))for i,c in enumerate(l['text'])]
            add(kind+'/'+mode,q)
    changed={k:sum(c.get('expected',{}).get('lines')!=c['request']['lines']for c in cases if c['request']['kind']==k and 'expected'in c)for k in KINDS};assert all(changed.values()),changed
    write_json(target,dict(cases=cases));print(len(cases),'existing Python final title rules;',changed,'positive;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
