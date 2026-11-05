"""Execute the original page-title statements through the two cross-lane passes.

The prefix ends before paragraph-title classification, without replacing that stage.
"""
import ast,copy,json,random
from dataclasses import asdict
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean

def profile_raw(p):
    r=asdict(p);r['style_support']=[dict(font=f,ratio=v)for f,v in p.style_support.items()];return r

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis import page_titles as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/title_analysis/page_titles.py'
    fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)and n.name=='_classify_page_titles')
    fn=copy.deepcopy(fn);angle=next(n for n in fn.body if isinstance(n,ast.For));stop=next(i for i,n in enumerate(angle.body)if isinstance(n,ast.For)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Name)and c.func.id=='_classify_paragraph_titles_in_lane'for c in ast.walk(n)))
    angle.body=angle.body[:stop]+ast.parse('collect_context(locals())').body
    ls=lines(q['lines']);indices={id(l):i for i,l in enumerate(ls)};contexts=[]
    def collect_context(v):
        lanes=[]
        for lane in v['lanes']:
            p=old._infer_lane_body_profile(lane)
            lanes.append(dict(left=lane.left,right=lane.right,is_span=lane.is_span,line_indices=[indices[id(l)]for l,_ in lane.lines],profile=profile_raw(p)))
        contexts.append(dict(angle=v['angle'],line_indices=[indices[id(l)]for l,_ in v['line_geometry']],median_height=v['median_height'],local_page_size=[v['local_page_width'],v['local_page_height']],lanes=lanes,physical_gaps=[dict(source_index=i,above=g[0],below=g[1])for i,g in v['physical_gaps'].items()],grid_title_suppressions=sorted(v['grid_title_suppressions']),document_title_bottom=v['document_title_bottom'],preserve_front_matter_boundaries=v['preserve_front_matter_boundaries']))
    body=q.get('document_body_profile');body=None if body is None else _DocumentBodyProfile(**dict(body,regular_fonts=frozenset(tuple(f)for f in body['regular_fonts'])))
    module=ast.Module(body=fn.body,type_ignores=[]);ast.fix_missing_locations(module)
    scope=dict(vars(old),lines=ls,page_size=tuple(q['page_size']),page_index=q['page_index'],container_bboxes=[tuple(b)for b in q['container_bboxes']],document_body_profile=body,collect_context=collect_context)
    exec(compile(module,str(path),'exec'),scope)
    return clean(dict(lines=[project(l)for l in ls],contexts=contexts))

def line(t,b,i,h=10.,font='Body',weight=400.,**kw):return dict(text=t,bbox=b,source_index=i,angle=0,chars=[],effective_height=h,em_height=h,font_signature=[font,0]if font else None,font_coverage=1.,dominant_font_weight=weight,**kw)
def main():
    target=ROOT/'build/migration/native-title-layout.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    body=dict(body_height=10.,body_weight=400.,regular_fonts=[['Body',0]],has_style_scale_repairs=False)
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('page_index',0);q.setdefault('container_bboxes',[]);q.setdefault('document_body_profile',None);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-document-auxiliary-canonical.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        for i,p in enumerate(c['expected']['pages']):add(c['label']+'/'+str(i),dict(lines=p['remaining_lines'],page_size=p.get('page_size',p.get('source',{}).get('page_size')),page_index=i,container_bboxes=[b['bbox']for b in p.get('fixed_blocks',[])],document_body_profile=c['expected']['body_profile']))
    def base():return dict(lines=[line('Original body paragraph text',[30.,y,370.,y+10.],i)for i,y in enumerate([140.,160.,180.,200.,220.,240.,260.,280.])],document_body_profile=body)
    for mode in ['anchor','multiline','shrink-boundary','fonts','low-coverage','weight-wall','off-center','cross-lane','second-band','fallback','tie','no-profile','empty','rotated','explicit','suppressed']:
        q=base();q['lines']+=[line('Original document title',[40.,30.,360.,50.],20,h=20.,font='Title',weight=700.)]
        if mode in ['multiline','shrink-boundary','fonts','low-coverage','weight-wall','cross-lane']:q['lines'].append(line('Original continuation',[80.,55.,320.,75.],21,h=20.,font='Title',weight=700.))
        if mode=='shrink-boundary':q['lines'][-1].update(effective_height=17.,em_height=17.)
        if mode in ['fonts','low-coverage']:q['lines'][-1]['font_signature']=['Other',0]
        if mode=='low-coverage':q['lines'][-1]['font_coverage']=.8
        if mode=='weight-wall':q['lines'][-1]['dominant_font_weight']=400.
        if mode=='off-center':q['lines'][-1]['bbox']=[0.,30.,100.,50.]
        if mode=='cross-lane':q['lines'][-1]['bbox']=[130.,55.,360.,75.]
        if mode=='second-band':q['document_body_profile']=dict(body,has_style_scale_repairs=True);q['lines']+=[line('Original secondary title',[80.,110.,320.,120.],21,font='Other'),line('Original author',[160.,125.,240.,135.],22)]
        if mode=='fallback':q['lines'][-1].update(effective_height=13.,em_height=13.)
        if mode=='tie':q['lines'].append(line('Original alternative',[40.,30.,360.,50.],21,h=20.,font='Title',weight=700.))
        if mode=='no-profile':q['document_body_profile']=None
        if mode=='empty':q['lines']=[]
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            q['page_size']=[600.,400.]
            for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        if mode=='explicit':q['lines'][-1].update(semantic_type='paragraph_title',explicit_section_title=True)
        if mode=='suppressed':q['lines'][-1]['title_suppressed']=True
        add('document/'+mode,q)
    for mode in ['front','no-keyword','no-author','lowercase-author','close-author','five-lines','metadata','nfkc-metadata','unicode-space','copyright-in-author','rotated']:
        q=base();q['page_index']=1;q['lines']+=[line('Original translated scientific title',[40.,50.,360.,60.],20),line('Original translated continued title',[70.,65.,330.,75.],21),line('AUTHOR Firstname',[120.,90.,280.,100.],22),line('Abstract: Original contents',[30.,120.,370.,130.],23),line('Keywords: Original',[30.,300.,250.,310.],24)]
        if mode=='no-keyword':q['lines'][-1]['text']='Original keywords without colon'
        if mode=='no-author':q['lines'][-3]['text']='Author lowercase'
        if mode=='lowercase-author':q['lines'][-3]['text']='author Firstname'
        if mode=='close-author':q['lines'][-3]['bbox']=[120.,80.,280.,90.]
        if mode=='five-lines':q['lines']+=[line('Original extra translated scientific title',[40.,80.+i*10.,360.,90.+i*10.],30+i)for i in range(3)]
        if mode in ['metadata','nfkc-metadata']:q['lines'][-3]['text']='１Original institution'if mode=='nfkc-metadata'else'1Original institution'
        if mode=='unicode-space':q['lines'][-3]['text']='AUTHOR\u001cFirstname'
        if mode=='copyright-in-author':q['lines'][-3]['text']='_AUTHOR Firstname'
        if mode=='rotated':
            for l in q['lines']:l['angle']=90
        add(mode,q)
    for mode in ['centered','italic','regular','first-page-small','near-image','inside','font-coverage','no-above','no-below','discontinuous']:
        q=base();q['page_index']=1;q['lines'].append(line('Original section',[140.,190.,260.,200.],20,h=10.,font='Italic'if mode=='italic'else'Body'));q['lines']=[l for l in q['lines']if l['source_index']not in [3]]
        for l in q['lines']:
            if l['source_index']>=4 and l['source_index']<20:l['bbox'][1]+=10.;l['bbox'][3]+=10.
        if mode=='italic':q['lines'][-1].update(font_signature=['Italic',64],bbox=[30.,190.,160.,200.])
        if mode=='first-page-small':q['page_index']=0;q['lines'][-1].update(effective_height=8.,em_height=8.)
        if mode=='near-image':q['container_bboxes']=[[100.,201.,300.,300.]]
        if mode=='inside':q['container_bboxes']=[[100.,185.,300.,205.]]
        if mode=='font-coverage':q['lines'][-1]['font_coverage']=.5
        if mode=='no-above':q['lines']=[l for l in q['lines']if l['bbox'][1]>=190.]
        if mode=='no-below':q['lines']=[l for l in q['lines']if l['bbox'][1]<=190.]
        if mode=='discontinuous':q['lines'][4]['bbox'][1]+=30.;q['lines'][4]['bbox'][3]+=30.
        add('cross/'+mode,q)
    rng=random.Random(827)
    for i in range(120):
        q=base();q['page_index']=i%3
        for l in q['lines']:
            h=rng.choice([8.,10.,12.,16.]);l.update(effective_height=h,em_height=h,font_signature=[rng.choice(['Body','Title','ABCDEF+Body']),rng.choice([0,64])],font_coverage=rng.choice([.5,.75,1.]),dominant_font_weight=rng.choice([None,400.,500.,700.]));l['bbox'][0]+=rng.choice([0.,40.,80.]);l['bbox'][2]-=rng.choice([0.,80.,160.])
        q['lines'].append(line('Original randomized title',[40.,30.,360.,50.],20,h=rng.choice([12.,16.,20.])))
        add('mixed/'+str(i),q)
    assert any(any(l['semantic_type']=='doc_title'for l in c.get('expected',{}).get('lines',[]))for c in cases)
    write_json(target,dict(cases=cases));print(len(cases),'old page-title layout prefixes;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
