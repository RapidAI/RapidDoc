"""Focused font-null/Unicode/exception and CPython anchor-set order contracts."""
import copy,random
from common import ROOT,write_json
from export_native_page_titles import reference
from export_native_title_layout import line
from export_native_geometry import lines
from export_native_late_image_text import project

def main():
    target=ROOT/'build/migration/native-title-contracts.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['classify_page']=True;c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    for font in ['', ' _- ', 'ABCDEF+', 'Body',None]:
        for mode in ['normal','inline','caption','structural']:
            ls=[line('Original complete body paragraph words',[30.,y,370.,y+10.],i,font='Body')for i,y in enumerate([100.,115.,130.,180.,195.,210.])]
            ls+=[line('Original title',[30.,150.,170.,160.],20,font=font)]
            ls=[project(l)for l in lines(ls)]
            if font is not None:ls[-1]['font_signature']=[font,0]
            if mode=='inline':ls[-1]['bbox'][2]=370.;ls.append(project(lines([line('Original body continuation',[30.,165.,370.,175.],21,font=font)])[0]))
            if mode=='caption':ls[-1].update(semantic_type='paragraph_title',explicit_section_title=True)
            if mode=='structural':ls[-1].update(semantic_type='paragraph_title',structural_title=True,explicit_section_title=True)
            add('font/'+repr(font)+'/'+mode,dict(lines=ls,page_size=[400.,600.],page_index=1,container_bboxes=[],caption_container_bboxes=[[20.,130.,200.,148.]]if mode=='caption'else[],document_body_profile=dict(body_height=10.,body_weight=400.,regular_fonts=[],has_style_scale_repairs=False)))
    for text in ['Abstract: native','abſtract：native','Ａｂｓｔｒａｃｔ：native','\u001c摘要：原始','Abstract no colon']:
        for institution in ['1Original institution','１Original institution','١Original institution','1original institution']:
            ls=[line('Original scientific translated document heading',[40.,50.,360.,60.],0),line('Original translated continuation heading',[50.,65.,350.,75.],1),line(institution,[100.,95.,300.,105.],2),line(text,[30.,125.,370.,135.],3),line('Keywords: original',[30.,200.,300.,210.],4)]
            add('metadata/'+text+'/'+institution,dict(lines=[project(l)for l in lines(ls)],page_size=[400.,600.],page_index=1,container_bboxes=[],document_body_profile=None))
    q=dict(lines=[project(l)for l in lines([line('Original paragraph',[30.,100.,370.,110.],0),line('Original heading',[30.,140.,170.,150.],1)])],page_size=[400.,600.],page_index=1,container_bboxes=[],document_body_profile=None)
    for field in ['effective_height','em_height','font_coverage']:
        r=copy.deepcopy(q);r['lines'][0][field]=None;add('old-error/'+field,r)
    traces=[];rng=random.Random(713)
    for mode in ['collisions','resize','discard','mixed']:
        value=set();steps=[]
        keys=[1,32,65,96,129,160]if mode=='collisions'else list(range(90))if mode=='resize'else [1,32,33,64,65,96,97,128]
        operations=[('add',i)for i in keys]
        if mode=='discard':operations+=[('discard',i)for i in keys[::2]]+[('add',i)for i in reversed(keys[::2])]
        if mode=='mixed':operations += [(rng.choice(['add','discard']),rng.randrange(4096))for _ in range(100)]
        for op,key in operations:
            getattr(value,op)(key);steps.append(dict(operation=op,key=key,order=list(value)))
        traces.append(dict(label=mode,steps=steps))
    write_json(target,dict(cases=cases,set_traces=traces));print(len(cases),'title contracts;',sum('error'in c for c in cases),'original errors; 4 set traces')
if __name__=='__main__':main()
