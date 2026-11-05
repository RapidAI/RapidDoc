"""Freeze complete original bar detector and graphic-caption mutations; no reference overwrite."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_tiled_tables import old_source
from export_native_bar_groups import base as bars
MUTABLE=('semantic_type','title_suppressed','paragraph_group')
def objects(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    out=[]
    for i,l in enumerate(copy.deepcopy(raw)):
        if l.get('font_signature')is not None:l['font_signature']=tuple(l['font_signature'])
        out.append(_LineItem(**dict(l,angle=l.get('angle',0),source_index=l.get('source_index',i))))
    return out
def snapshot(raw,lines,before):
    result=copy.deepcopy(raw)
    for r,l,prev in zip(result,lines,before):
        for k,v in zip(MUTABLE,prev):
            if getattr(l,k)!=v:r[k]=clean(getattr(l,k))
    return result
def caption_reference(request):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _graphic_caption_line_indices_to_preserve
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _GraphicCandidate
    raw=request['lines'];lines=objects(raw);before=[tuple(getattr(l,k)for k in MUTABLE)for l in lines];protected=_graphic_caption_line_indices_to_preserve(lines,[_GraphicCandidate(tuple(b),-1)for b in request['candidate_bboxes']],request['median_height']);return clean(dict(lines=snapshot(raw,lines,before),protected_source_indices=protected))
def reference(source,em):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_native_bar_graphics
    s=old_source(source);s.lines=objects(source.get('lines',[]));s.compound_baseline_cache={};s.image_bboxes=source.get('image_bboxes',[]);s.form_bboxes=source.get('form_bboxes',[]);before=[tuple(getattr(l,k)for k in MUTABLE)for l in s.lines];boxes=_detect_native_bar_graphics(s,em);result=copy.deepcopy(source)
    if 'lines'in result:result['lines']=snapshot(source['lines'],s.lines,before)
    return clean(dict(source=result,bar_bboxes=boxes))
def caption_base():
    return [dict(text='Fig 1: Example Caption',bbox=[50.,50.,180.,60.],angle=0,source_index=0,font_signature=['ABCDEF+Serif',0],dominant_font_weight=400),dict(text='Second caption line.',bbox=[50.,65.,160.,75.],angle=0,source_index=1,font_signature=['ABCDEF+Serif',0],dominant_font_weight=400)]
def main():
    caption_target=ROOT/'build/migration/native-graphic-captions.json';target=ROOT/'build/migration/native-bar-graphics.json'
    if caption_target.exists()or target.exists():raise FileExistsError('immutable graphic references exist')
    captions=[];cases=[]
    def cap(label,lines=None,boxes=None,em=10.):
        q=clean(dict(lines=lines if lines is not None else caption_base(),candidate_bboxes=boxes if boxes is not None else[[40.,90.,220.,200.]],median_height=em));captions.append(dict(label=label,request=q,expected=caption_reference(q)))
    cap('wrapped-above');cap('below',boxes=[[40.,0.,220.,40.]]);cap('far',boxes=[[300.,300.,400.,400.]])
    texts=['Fig 1: Caption','Figure 2. Example','FIG.3 A Title','fig1a','fig1and description','Fig 1 shows this','Fig 1: lower text','Fig 1 和其他','Fig 1 (a), next','Fig 1 (a): title','Fig 1:','Ｆｉｇ １: Title','FİG 1: Title','fıg 1: Title','Fig١: Title','Fig ١: Title','Fig１: Title','Fig 1.2: Title','Fig 1 - A2: Title','Fig 1A: Title','Fig 1A lower title','Fig 1 Provides results','Fig 1 Ⅳ Results','Fig\n1: Title','Figure 1\u00a0: Title','图1 文字','Tab 1: Title']
    for text in texts:l=caption_base();l[0]['text']=text;cap('syntax-'+repr(text),l)
    for text in ['two words','single','axis 12','source: source text','Sources：source text','12%','١٢','New Fig 2','Fig 2: Another','_two words','éwords here','wordsé here','ab_cd more','two words!','two words。']:
        l=caption_base();l[1]['text']=text;cap('tail-'+repr(text),l)
    for signature in [None,['GHIJKL+Serif',1],['Ser_if',0],['Ser-if',0],['Serif',1],['Other',0],['ABCDEF+SERIF',0],['ABCDEF+',0],['Straße',0]]:
        l=caption_base();l[1]['font_signature']=signature;cap('family-'+str(signature),l)
    for w in [None,0,300,301,499,500]:l=caption_base();l[1]['dominant_font_weight']=w;cap('weight-'+str(w),l)
    for x in [46.9999,47.,47.0001,53.,53.0001]:l=caption_base();l[1]['bbox'][0]=x;cap('indent-'+str(x),l)
    for y in [67.4999,67.5,67.5001,50.,49.]:l=caption_base();l[1]['bbox'][1]=y;l[1]['bbox'][3]=y+10.;cap('gap-'+str(y),l)
    for group in [None,-5,-1,0,10]:l=caption_base();l[0]['paragraph_group']=group;cap('existing-group-'+str(group),l)
    l=caption_base();l[1]['source_index']=0;cap('duplicate-source',l)
    for angle in [0,90,180,270,15]:l=caption_base();l[1]['angle']=angle;cap('angle-'+str(angle),l)
    rng=random.Random(261111)
    for i in range(70):
        l=caption_base();rng.shuffle(l);l[rng.randrange(2)]['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);cap('geometry-'+str(i),l)
    def add(label,s,em=10.):s=clean(copy.deepcopy(s));cases.append(dict(label=label,request=dict(source=s,em=em),expected=reference(s,em)))
    for h in [True,False]:
        for compound in [False,True]:add(f'base-{h}-{compound}',bars(h,compound))
    s=bars();s['image_bboxes']=[[140.,100.,210.,170.]];add('image-extension',s)
    for image in [[0.,0.,400.,600.],[140.,100.,210.,170.],[140.,185.,210.,195.],[300.,300.,350.,350.]]:s=bars();s['image_bboxes']=[image];add('image-'+str(image),s)
    for text in ['ordinary label','Sources: reference','Sources：reference','来源：参考','注：说明','Figure 2: Caption','nine words in this very long and elaborate chart label']:
        s=bars();s['lines'].append(dict(text=text,bbox=[100.,172.,160.,182.],angle=30,source_index=8,chars=[]));add('label-'+text,s)
    for b in [[80.,172.,110.,190.],[80.,190.,110.,230.],[80.,172.,250.,190.]]:
        for note in [False,True]:
            s=bars();s['path_infos'].append(dict(s['path_infos'][0],bbox=b,segment_count=25))
            if note:s['lines'].append(dict(text='Source: source text',bbox=[80.,175.,140.,185.],source_index=8,chars=[]))
            add('outline-'+str(b)+str(note),s)
    for height in [0.,12.5,12.5001,30.]:
        for repaired in [False,True]:s=bars();s['lines'].append(dict(text='label',bbox=[30.,110.,49.,120.],source_index=8,chars=[],effective_height=height,em_height=10.,style_scale_repaired=repaired));add('height-'+str(height)+str(repaired),s)
    s=bars();s['lines']+=caption_base();add('caption-mutations',s)
    # A real occupied grid spans the candidate; admission is unaffected by numeric labels outside it.
    s=bars();s['drawing_lines']+=[dict(bbox=[50.,y,190.,y],orientation='horizontal',width=1.)for y in [100.,130.,160.,170.]]+[dict(bbox=[x,100.,x,170.],orientation='vertical',width=1.)for x in [50.,120.,190.]]
    for y in [110.,140.,162.]:
        for x in [65.,140.]:s['lines'].append(dict(text='cell',bbox=[x,y,x+20.,y+5.],angle=0,source_index=len(s['lines']),chars=[]))
    add('occupied-grid',s)
    for i in range(90):
        s=bars(bool(i%2),bool(i%3==0));rng.shuffle(s['path_infos']);rng.shuffle(s['lines'])
        s['lines'].append(dict(text=rng.choice(['label','Fig 1: A Title','Source: other','two words.']),bbox=[20.,90.+i%10,45.,100.+i%10],source_index=8,angle=rng.choice([0,15,90]),chars=[]));add('combined-'+str(i),s)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/annotation_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/typography.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    sources={p:digest((ROOT/p).read_bytes())for p in refs};write_json(caption_target,dict(sources=sources,cases=captions));write_json(target,dict(sources=sources,cases=cases));print('Frozen captions',len(captions),'protected',sum(bool(c['expected']['protected_source_indices'])for c in captions),'mutated',sum(c['request']['lines']!=c['expected']['lines']for c in captions));print('Frozen full bars',len(cases),'positive',sum(bool(c['expected']['bar_bboxes'])for c in cases),'mutated',sum(c['request']['source']!=c['expected']['source']for c in cases))
if __name__=='__main__':main()
