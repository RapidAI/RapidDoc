"""Freeze complete original page-title classification; no title probes are stubbed."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_title_layout import line
from export_native_inline_clusters import clean

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.page_titles import _classify_page_titles
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile,_DocumentTitleProfile,_TitleStylePrototype
    ls=lines(q['lines']);b=q.get('document_body_profile');t=q.get('document_title_profile')
    b=None if b is None else _DocumentBodyProfile(**dict(b,regular_fonts=frozenset(tuple(f)for f in b['regular_fonts'])))
    t=None if t is None else _DocumentTitleProfile(tuple(_TitleStylePrototype(**p)for p in t['prototypes']))
    _classify_page_titles(ls,tuple(q['page_size']),page_index=q['page_index'],container_bboxes=[tuple(b)for b in q['container_bboxes']],caption_container_bboxes=[tuple(b)for b in q.get('caption_container_bboxes',[])],document_body_profile=b,document_title_profile=t)
    return clean(dict(lines=[project(l)for l in ls]))

def main():
    target=ROOT/'build/migration/native-page-titles.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['classify_page']=True;q.setdefault('page_size',[400.,600.]);q.setdefault('page_index',1);q.setdefault('container_bboxes',[]);q.setdefault('document_body_profile',None);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    for c in json.loads((ROOT/'build/migration/native-title-layout.json').read_text(encoding='utf-8'))['cases']:add(c['label'],c['request'])
    body=dict(body_height=10.,body_weight=400.,regular_fonts=[['Body',0]],has_style_scale_repairs=False)
    proto=dict(font_family='title',font_flags=0,height_ratio=1.4,weight=700.,alignment='left',anchor_offset=0.,support_count=3,support_pages=2)
    def base():return dict(lines=[line('Original normal body paragraph words',[30.,y,370.,y+10.],i)for i,y in enumerate([90.,105.,120.,180.,195.,210.,270.,285.,300.])]+[line('Section original title',[30.,150.,180.,164.],20,h=14.,font='Title',weight=700.)],document_body_profile=body,document_title_profile=dict(prototypes=[proto]))
    for mode in ['prototype','prototype-size','prototype-family','prototype-weight','prototype-anchor','prototype-low-coverage','full-width-inline','mixed-inline','regular-pitch','sentence','caption','unknown','null-font','wrapped','terminal-neighbor','near-container','body-font','front-cover','repair','visual-row-compatible','visual-row-mixed','visual-row-body','single-glyph','suppressed','explicit']:
        q=base();l=q['lines'][-1]
        if mode=='prototype-size':l.update(effective_height=10.,em_height=10.)
        if mode=='prototype-family':l['font_signature']=['TitleBold',0]
        if mode=='prototype-weight':l['dominant_font_weight']=400.
        if mode=='prototype-anchor':l['bbox']=[180.,150.,330.,164.]
        if mode=='prototype-low-coverage':l['font_coverage']=.6
        if mode=='full-width-inline':l.update(bbox=[30.,150.,370.,160.],effective_height=10.,em_height=10.);q['lines'].append(line('Original continuation body words',[30.,165.,370.,175.],21))
        if mode=='mixed-inline':l.update(bbox=[30.,150.,370.,160.],effective_height=10.,em_height=10.,font_coverage=.5);q['lines'].append(line('Original continuation body words',[30.,165.,370.,175.],21))
        if mode=='regular-pitch':l.update(bbox=[30.,135.,370.,145.],effective_height=10.,em_height=10.,font_signature=['Body',0],dominant_font_weight=400.);q['document_title_profile']=None
        if mode=='sentence':q['lines']=[line(t,[30.,100.+i*15.,370.,110.+i*15.],i,font='Italic')for i,t in enumerate(['Original sentence includes many words, and remains body','Original sentence contains enough words for this section','Original concluding sentence has several ordinary words.'])];q['document_title_profile']=None
        if mode=='caption':q['caption_container_bboxes']=[[20.,110.,210.,148.]]
        if mode=='unknown':l['dominant_font_weight']=None
        if mode=='null-font':l['font_signature']=None
        if mode in ['wrapped','terminal-neighbor']:q['lines'].append(line('Original continued title.'if mode=='terminal-neighbor'else'Original continued title',[35.,168.,185.,182.],21,h=14.,font='Title',weight=700.))
        if mode=='near-container':q['container_bboxes']=[[20.,165.,210.,230.]]
        if mode=='body-font':l['font_signature']=['Body',0]
        if mode=='front-cover':q['page_index']=0;q['lines'].append(line('Original document heading',[40.,20.,360.,45.],21,h=25.,font='Cover',weight=700.))
        if mode=='repair':q['document_body_profile']=dict(body,has_style_scale_repairs=True)
        if mode.startswith('visual-row'):
            l.update(visual_row_id=9,split_from_row=True);q['lines'].append(line('Original same visual row',[190.,150.,350.,164.],21,h=14.,font='Body'if mode=='visual-row-mixed'else'Title',weight=400. if mode=='visual-row-mixed'else 700.,visual_row_id=9,split_from_row=True))
            if mode=='visual-row-body':q['lines'][-1].update(semantic_type='text')
        if mode=='single-glyph':l.update(bbox=[30.,150.,40.,160.],effective_height=10.,em_height=10.,median_glyph_width=10.)
        if mode=='suppressed':l['title_suppressed']=True
        if mode=='explicit':l.update(semantic_type='paragraph_title',explicit_section_title=True,structural_title=True,title_band_id=20)
        q['lines'].sort(key=lambda l:(l['bbox'][1],l['bbox'][0]));add('full/'+mode,q)
    changed=sum('expected'in c and c['expected']['lines']!=c['request']['lines']for c in cases)
    assert changed>20
    write_json(target,dict(cases=cases));print(len(cases),'complete old page-title cases;',changed,'changed;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
