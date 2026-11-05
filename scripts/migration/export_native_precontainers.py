"""Freeze original correspondence and appendix-only pre-container classifiers."""
import copy
from common import ROOT,write_json
from export_form_runtime import typed
from export_native_late_image_text import project
from export_native_image_chain import line
from export_native_table_page_policies import empty

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.auxiliary_text import _classify_first_page_correspondence_footnotes
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.structural import _classify_bold_numbered_heading_rows
    s=typed(q['source'])
    if q['kind']=='correspondence':_classify_first_page_correspondence_footnotes(s)
    else:_classify_bold_numbered_heading_rows(s.lines,s.page_size,[tuple(b)for b in q['container_bboxes']],appendix_only=True)
    out=copy.deepcopy(q['source']);out['lines']=[project(l)for l in s.lines];return out

def main():
    target=ROOT/'build/migration/native-precontainers.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,source,kind='appendix',boxes=()):
        source=copy.deepcopy(source);source['lines']=[project(l)for l in typed(source).lines]
        q=dict(kind=kind,source=source,container_bboxes=list(boxes));cases.append(dict(label=label,request=q,expected=reference(q)))
    def heading(t='A Original appendix title',b=(30.,150.,260.,160.),i=0):
        l=line(t,list(b),i,font_coverage=1.,dominant_font_weight=700.);l['font_signature']=['Body-Bold',0];return l
    for t in ['A Original appendix title','A','A A B','A 标题测试','A Title.','A Title:','A—Title','A - Title','A\u001cTitle','B Original appendix title with many words','1 Original numbered title','Ａ Fullwidth appendix','a Lowercase label','A 2026 original title','A x = y']:
        s=empty();s['lines']=[heading(t)];add('text/'+t,s)
    for mode in ['split','no-companion','far','y-miss','all-caps','regular','coverage','wide','container','already','continuation','digit-continuation','new-heading','font-mismatch','same-row','suppressed','repaired-height']:
        s=empty();s['lines']=[heading('A',(30.,150.,38.,160.)),heading('Original appendix title with many words',(45.,150.,260.,160.),1)]
        if mode=='no-companion':s['lines']=s['lines'][:1]
        if mode=='far':s['lines'][1]['bbox'][0]=70.
        if mode=='y-miss':s['lines'][1]['bbox'][1:4:2]=[158.,168.]
        if mode=='all-caps':s['lines'][1]['text']='A B C'
        if mode=='regular':s['lines'][1].update(font_signature=['Body',0],dominant_font_weight=400.)
        if mode=='coverage':s['lines'][1]['font_coverage']=.749
        if mode=='wide':s['lines'][1]['bbox'][2]=490.
        if mode=='already':s['lines'][0].update(title_band_id=9,structural_title=True)
        if mode in ['continuation','digit-continuation','new-heading','font-mismatch','same-row']:
            s['lines'].append(heading('2026 original continuation'if mode=='digit-continuation'else'continued original title',(45. if mode!='new-heading'else 30.,162. if mode!='same-row'else 150.,240.,172. if mode!='same-row'else 160.),2))
            if mode=='new-heading':s['lines'][-1]['text']='B New heading'
            if mode=='font-mismatch':s['lines'][-1]['font_signature']=['Other-Bold',0]
        if mode=='suppressed':s['lines'][0]['title_suppressed']=True
        if mode=='repaired-height':s['lines'][0].update(style_scale_repaired=True,em_height=8.,effective_height=50.)
        add(mode,s,boxes=[[20.,140.,270.,180.]]if mode=='container'else [])
    for mode in ['positive','five','weight','coverage','caption','formula','inline','container','same-baseline','one-follower','far-above','far-below','digit','bullet','punctuation','zero-weight-follower']:
        s=empty();s['lines']=[heading('Original repaired heading',(30.,100.,210.,110.))]
        s['lines'] += [line('Original complete body paragraph',[30.,y,330.,y+10.],i+1,style_scale_repaired=True,em_height=10.,source_bbox=[30.,y-20.,330.,y+30.],dominant_font_weight=400.,font_coverage=1.)for i,y in enumerate([120.,140.,160.,180.,200.,220.])]
        if mode=='five':s['lines']=s['lines'][:-1]
        if mode=='weight':s['lines'][0]['dominant_font_weight']=599.
        if mode=='coverage':s['lines'][0]['font_coverage']=.399
        if mode=='caption':s['lines'][0]['caption_start']=True
        if mode=='formula':s['lines'][0]['formula_candidate_only']=True
        if mode=='inline':s['lines'][0]['inline_math_regions']=[[30.,100.,40.,110.]]
        if mode=='same-baseline':s['lines'].append(line('same baseline',[250.,100.,310.,110.],20))
        if mode=='one-follower':
            for l in s['lines'][2:]:l['dominant_font_weight']=600.
        if mode=='far-above':s['lines'].append(line('previous',[30.,20.,330.,30.],20))
        if mode=='far-below':
            for l in s['lines'][1:]:l['bbox'][1]+=20.;l['bbox'][3]+=20.
        if mode=='digit':s['lines'][0]['text']='٣ Original heading'
        if mode=='bullet':s['lines'][0]['text']='• Original heading'
        if mode=='punctuation':s['lines'][0]['text']='Original heading.'
        if mode=='zero-weight-follower':
            for l in s['lines'][1:]:l['dominant_font_weight']=0.
        add('extreme/'+mode,s,boxes=[[20.,90.,220.,115.]]if mode=='container'else [])
    for mode in ['positive','same-field','no-body','small-body','large-contact','numbered-break','gap','misaligned','rotated','semantic','two-columns','fullwidth-number','three-fields']:
        s=empty();s['page_size']=[400.,600.];s['lines']=[line('Original long body text in the document',[30.,y,370.,y+10.],i)for i,y in enumerate([150.,180.,210.,240.])]
        s['lines'] += [line('Corresponding author: original',[30.,440.,250.,447.],10),line('Received: 2026',[30.,450.,250.,457.],11)]
        if mode=='same-field':s['lines'][-1]['text']='Email: original@example.org'
        if mode=='no-body':s['lines']=s['lines'][-2:]
        if mode=='small-body':
            for l in s['lines'][:4]:l['effective_height']=7.
        if mode=='large-contact':s['lines'][-2]['effective_height']=9.
        if mode=='numbered-break':s['lines'][-1]['text']='1. Received: 2026'
        if mode=='fullwidth-number':s['lines'][-1]['text']='１. Received: 2026'
        if mode=='gap':s['lines'][-1]['bbox']=[30.,470.,250.,477.]
        if mode=='misaligned':s['lines'][-1]['bbox']=[200.,450.,390.,457.]
        if mode=='rotated':s['lines'][-2]['angle']=90
        if mode=='semantic':s['lines'][-2]['semantic_type']='paragraph_title'
        if mode=='two-columns':
            for l in s['lines'][:4]:l['bbox'][2]=180.
            s['lines'] += [line('Right original body column',[220.,y,370.,y+10.],20+i)for i,y in enumerate([150.,180.,210.,240.])]
        if mode=='three-fields':s['lines'].append(line('Accepted: 2026',[30.,460.,250.,467.],12))
        for l in s['lines']:
            if l['source_index']>=10 and l['source_index']<20:l.setdefault('effective_height',7.);l['effective_height']=7. if l['effective_height']==10. else l['effective_height']
        add('contact/'+mode,s,'correspondence')
    write_json(target,dict(cases=cases));print(len(cases),'old pre-container classifier cases;',sum(c['expected']!=c['request']['source']for c in cases),'mutating scenes')
if __name__=='__main__':main()
