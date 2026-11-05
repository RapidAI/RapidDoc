"""Freeze complete existing Python final-page structural-title classifiers."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line
BODY=dict(body_height=10.,body_weight=400.,regular_fonts=[['Body',0]],has_style_scale_repairs=False)
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis import structural as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _DocumentBodyProfile
    ls=lines(q['lines']);body=q.get('document_body_profile');body=None if body is None else _DocumentBodyProfile(**dict(body,regular_fonts=frozenset(tuple(f)for f in body['regular_fonts'])))
    kwargs=dict(container_bboxes=[tuple(b)for b in q['container_bboxes']],document_body_profile=body)
    if q['kind']=='body':kwargs['page_index']=q['page_index']
    if q['kind']=='promote':kwargs.update(page_index=q['page_index'],title_candidate_source_indices=set(q['title_candidate_source_indices']))
    fn=dict(explicit=old._classify_explicit_section_titles,reset=old._classify_inline_typography_reset_titles,body=old._classify_body_height_section_titles,promote=old._promote_noninitial_document_title_band)[q['kind']]
    fn(ls,tuple(q['page_size']),**kwargs);return clean(dict(lines=[project(l)for l in ls]))
def main():
    target=ROOT/'build/migration/native-page-structure.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('container_bboxes',[]);q.setdefault('page_index',1);q.setdefault('document_body_profile',BODY);q['lines']=[project(l)for l in lines(q['lines'])];q.setdefault('title_candidate_source_indices',[l['source_index']for l in q['lines']if l['semantic_type']=='paragraph_title']);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-final-formula.json').read_text(encoding='utf-8'))
    for i,c in enumerate(f['cases']):
        if i%13:continue
        p=c['request']['page'];ls=c['expected']['remaining_lines'];size=p.get('page_size',p.get('source',{}).get('page_size'))
        for kind in ['explicit','reset','body','promote']:add('final/'+c['label']+'/'+kind,dict(lines=ls,page_size=size,kind=kind))
    def base():return dict(lines=[line('Original ordinary complete body words',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.,320.,340.])])
    for kind in ['explicit','reset','body','promote']:
        for mode in ['positive','no-profile','zero-profile','container','rotated','font','low-coverage','suppressed','margin','unicode','terminal','too-large','continuation','gap','first-page','wrong-number','numeric-label','lower-label','cjk','side-title','style-repair']:
            q=base();q['kind']=kind
            if kind=='explicit':q['lines'] += [line('2 Methods',[30.,230.,160.,240.],30)]
            if kind=='reset':q['lines']=[line('Tail words',[30.,170.,100.,180.],0),line('Original heading',[30.,184.,190.,194.],1,font='Heading'),line('Original ordinary complete body words',[45.,198.,370.,208.],2)]
            if kind=='body':q['lines']=[line('Heading',[30.,100.,90.,110.],0)]+[line('Original ordinary complete body words',[30.,y,370.,y+10.],i+1)for i,y in enumerate([114.,128.,142.])]+[line('Another',[30.,190.,90.,200.],4)]+[line('Original ordinary complete body words',[30.,y,370.,y+10.],i+5)for i,y in enumerate([204.,218.,232.])]
            if kind=='promote':q['lines'] += [line('Original document heading',[80.,190.,320.,205.],30,h=15.,font='Title',semantic_type='paragraph_title'),line('Original second heading',[100.,212.,300.,227.],31,h=15.,font='Title',semantic_type='paragraph_title')]
            candidate=q['lines'][-1]if kind=='explicit'else q['lines'][1]if kind=='reset'else q['lines'][0]if kind=='body'else q['lines'][-2]
            if mode=='no-profile':q['document_body_profile']=None
            if mode=='zero-profile':q['document_body_profile']=dict(BODY,body_height=0.)
            if mode=='container':q['container_bboxes']=[[20.,80.,380.,250.]]
            if mode=='font':candidate['font_signature']=['Other',0]
            if mode=='low-coverage':candidate['font_coverage']=.3
            if mode=='suppressed':candidate['title_suppressed']=True
            if mode=='margin':candidate['bbox'][1]=2.;candidate['bbox'][3]=12.
            if mode=='unicode':candidate['text']='２ 方法'
            if mode=='terminal':candidate['text']+='.'
            if mode=='too-large':candidate.update(effective_height=30.,em_height=30.)
            if mode=='continuation':q['lines'].append(line('Methods',[40.,241.,100.,251.],40))
            if mode=='gap':candidate['bbox'][1]+=15.;candidate['bbox'][3]+=15.
            if mode=='first-page':q['page_index']=0
            if mode=='wrong-number':candidate['text']='2026 Results'
            if mode=='numeric-label':candidate['text']='2 3 Results'
            if mode=='lower-label':candidate['text']='2 methods'
            if mode=='cjk':candidate['text']='参考文献'
            if mode=='side-title':candidate['bbox']=[30.,190.,150.,205.]
            if mode=='style-repair':candidate.update(style_scale_repaired=True,em_height=10.)
            if mode=='rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                q['page_size']=[600.,400.]
                for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
            add(kind+'/'+mode,q)
    changed={k:sum(c.get('expected',{}).get('lines')!=c['request']['lines']for c in cases if c['request']['kind']==k and 'expected'in c)for k in ['explicit','reset','body','promote']};assert all(changed.values()),changed
    write_json(target,dict(cases=cases));print(len(cases),'existing Python page-structure cases;',changed,'positive;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
