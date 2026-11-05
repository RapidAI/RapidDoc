"""Complete old caption-supported graph entry with real source tables/code barriers."""
import copy,json
from common import ROOT,digest,write_json
from export_form_runtime import typed
from export_native_inline_clusters import clean
from export_native_table_page_policies import empty
from export_native_image_chain import line
def reference(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _build_caption_supported_graphics
    s=typed(raw);fields=['semantic_type','paragraph_group','title_suppressed','median_glyph_width','explicit_section_title'];before=[[getattr(l,k)for k in fields]for l in s.lines];blocks,claimed=_build_caption_supported_graphics(s);source=copy.deepcopy(raw)
    for r,l,prev in zip(source['lines'],s.lines,before):
        for k,v in zip(fields,prev):
            if getattr(l,k)!=v:r[k]=clean(getattr(l,k))
    return clean(dict(source=source,caption_graphics=blocks,claimed_line_indices=sorted(claimed)))
def base():
    s=empty();s['page_size']=[400.,600.];s['image_bboxes']=[[40.,160.,360.,330.]];s['lines']=[line('Figure 1: Original photo',[40.,350.,360.,360.],0)]
    for i in range(3):s['lines'].append(line('native label',[60.+i*80,190.,110.+i*80,200.],i+1))
    for i in range(4):s['lines'].append(line('Original body text is a separate paragraph',[25.,420.+i*20,375.,430.+i*20],100+i))
    return s
def main():
    target=ROOT/'build/migration/native-caption-images.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    for mode in ['photo','fig-dot','fig-space-only','no-figure','title-floor','nearby-code','body-barrier','side-card','parallel-captions','white-fill','full-page-form']:
        s=base()
        if mode=='fig-dot':s['lines'][0]['text']='Fig. 1: Original photo'
        if mode=='fig-space-only':s['lines'][0]['text']='Fig 1: Original photo'
        if mode=='no-figure':s['lines'][0]['text']='普通正文'
        if mode=='title-floor':s['lines'].append(dict(line('Centered display heading',[40.,130.,360.,145.],200),effective_height=20.))
        if mode=='nearby-code':
            from export_native_code_containers import base as code_base
            code=code_base('filled')['source'];s['path_infos']+=code['path_infos'];s['lines']+=code['lines']
            for i,l in enumerate(s['lines']):l['source_index']=i
        if mode=='body-barrier':s['lines'].append(line('These native body words block a preceding image expansion',[30.,340.,370.,350.],300))
        if mode=='side-card':
            s['image_bboxes']=[[10.,100.,180.,300.]];s['lines']=[line('Figure 1',[190.,160.,230.,170.],0,font_coverage=1.)]
            for l in s['lines']:l['font_signature']=['Body',0]
            for i in range(3):s['lines'].append(dict(line('original side caption words',[240.,152.+i*12,380.,162.+i*12],i+1),font_signature=['Body',0]))
            s['drawing_lines']=[dict(bbox=b,width=.5,orientation=o)for b,o in [([185.,150.,385.,150.],'horizontal'),([185.,180.,385.,180.],'horizontal'),([185.,150.,185.,180.],'vertical'),([385.,150.,385.,180.],'vertical')]]
        if mode=='parallel-captions':s['image_bboxes']=[[20.,160.,180.,330.],[220.,160.,380.,330.]];s['lines'][0]['bbox']=[20.,350.,180.,360.];s['lines'].append(line('Figure 2: Second photo',[220.,350.,380.,360.],301))
        if mode=='white-fill':s['path_infos']=[dict(bbox=[0.,0.,400.,340.],form_depth=0,segment_count=5,source_index=0,fill_visible=True,stroke_visible=False,fill_rgba=[255,255,255,255],rectangle_bboxes=[])]
        if mode=='full-page-form':
            s['form_bboxes']=[[0.,0.,400.,600.]];s['retained_page_forms']=[[0.,0.,400.,600.]]
            for i,l in enumerate(s['lines']):l['chars']=[dict(char='X',char_idx=i,bbox=l['bbox'])]
            s['form_member_sources']=[dict(bbox=[0.,0.,400.,600.],source_indices=list(range(len(s['lines']))))]
        cases.append(dict(label=mode,request=copy.deepcopy(s),expected=reference(s)))
    refs={str(p.relative_to(ROOT)):digest(p.read_bytes())for p in [ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py']};write_json(target,dict(sources=refs,cases=cases));print(len(cases),'whole caption-supported source branches',sum(bool(c['expected']['caption_graphics'])for c in cases),'positive scenes')
if __name__=='__main__':main()
