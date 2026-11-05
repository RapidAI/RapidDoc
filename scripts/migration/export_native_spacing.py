"""Freeze complete existing spacing preparation/application, including real glyph evidence."""
import copy,json,random,math
from types import SimpleNamespace
from dataclasses import asdict
from common import ROOT,write_json
from export_native_inline_clusters import clean
def reference(q,kind):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.spacing import prepare_spacing_lines,apply_spacing_lines,PDFTextSpacingLine
    if kind=='prepare':return clean([asdict(l)for l in prepare_spacing_lines([SimpleNamespace(**dict(dict(formula_candidate_only=False,chars=[]),**l))for l in q['lines']],None if q.get('space_before')is None else set(q['space_before']))])
    bs=copy.deepcopy(q['blocks']);apply_spacing_lines(bs,[PDFTextSpacingLine(**dict(l,bbox=tuple(l['bbox']),boundaries=tuple(l['boundaries']),suppressed_boundaries=tuple(l.get('suppressed_boundaries',[]))))for l in q['spacing_lines']],tuple(q['page_size']));return clean(bs)
def main():
    target=ROOT/'build/migration/native-spacing.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q,kind):
        q=clean(copy.deepcopy(q));c=dict(label=label,kind=kind,request=q)
        try:c['expected']=reference(q,kind)
        except(ValueError,TypeError,KeyError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    for n,c in enumerate(source['cases']):
        if 'expected'not in c or n%3:continue
        ls=c['expected']['lines']
        for mode in [None,[],[1,3,5,8,13]]:add(f'source/{n}/{mode}',dict(lines=ls,space_before=mode),'prepare')
    def char(t,i,x=0,y=0,**kw):
        c=dict(char=t,char_idx=i,bbox=[x,y,x+5,y+10],tight_bbox=[x,y,x+5,y+10],font=dict(name='Body',flags=0,size=10.),writing_angle=0.,origin=[x,y+8],text_object_id=4);c.update(kw);return c
    def line(cs,n=0,**kw):return dict(bbox=[0.,0.,400.,10.],source_index=n,chars=cs,**kw)
    for mode in ['base','numeric','mono','scaled','large','small-script','missing-origin','baseline','wrong-angle','font-string','non-consecutive','zero-ink','reversed-ink','bool-source']:
        cs=[char('a',0),char('b',1,8),char('c',2,14)]
        if mode=='numeric':cs[0]['char']='1';cs[1]['char']='2'
        if mode=='mono':cs[0]['font']['flags']=1
        if mode=='scaled':
            for c in cs:c['font']['size']=1.
        if mode=='large':cs[1]['bbox'][3]=cs[1]['tight_bbox'][3]=40.
        if mode=='small-script':cs[1]['font']['size']=7.
        if mode=='missing-origin':
            for c in cs:c.pop('origin')
        if mode=='baseline':cs[1]['origin'][1]+=10.
        if mode=='wrong-angle':cs[1]['writing_angle']=1.
        if mode=='font-string':cs[0]['font']['flags']='1'
        if mode=='non-consecutive':cs[1]['char_idx']=8
        if mode=='zero-ink':cs[0]['tight_bbox'][2]=0.
        if mode=='reversed-ink':cs[0]['tight_bbox'][2]=-1.
        if mode=='bool-source':cs[0]['char_idx']=False;cs[1]['char_idx']=True
        for angle in [0,90,180,270]:
            v=copy.deepcopy(cs)
            if angle:
                for c in v:
                    c['writing_angle']+=math.radians(angle)
                    def rotate(b):
                        x0,y0,x1,y1=b
                        return {90:[-y1,x0,-y0,x1],180:[-x1,-y1,-x0,-y0],270:[y0,-x1,y1,-x0]}[angle]
                    c['bbox']=rotate(c['bbox']);c['tight_bbox']=rotate(c['tight_bbox'])
                    if 'origin'in c:
                        p=rotate(c['origin']*2);c['origin']=p[:2]
            for override in [None,[],[1]]:add(f'tight/{mode}/{angle}/{override}',dict(lines=[line(v)],space_before=override),'prepare')
    for content in ['中文','中，','，中','😀中','ﬁx','fix','ffix','stx']:
        if '中'in content:cs=[char(content[0],0),char(' ',1,5,is_generated=True),char(content[1],2,6)]
        else:
            cs=[char(t,i)for i,t in enumerate(content[:-1])];cs+=[char(' ',len(cs),2),char(content[-1],len(cs)+1,5)]
        for mode in ['base','not-generated','object','font','gap','origin','geometry','source','reordered','candidate']:
            v=copy.deepcopy(cs)
            if mode=='not-generated':v[-2]['is_generated']=False
            if mode=='object':v[-1]['text_object_id']=7
            if mode=='font':v[-1]['font']['name']='Different'
            if mode=='gap':v[-1]['tight_bbox'][0]+=20.;v[-1]['tight_bbox'][2]+=20.
            if mode=='origin':v[-1]['origin'][1]+=10.
            if mode=='geometry':v[0]['tight_bbox'][2]+=1.
            if mode=='source':v[-1]['char_idx']+=3
            if mode=='reordered':v.reverse()
            add(f'suppress/{content}/{mode}',dict(lines=[line(v,formula_candidate_only=mode=='candidate')]),'prepare')
    add('continuation',dict(lines=[line([char('inter-',0)],2),line([char('national',1)],3),line([char('other',2)],5)]),'prepare')
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.common import _normalize_match_fragment
    inputs=json.loads((ROOT/'build/migration/native-inline-output.json').read_text(encoding='utf-8'))['cases']
    for n,c in enumerate(inputs):
        q=c['request'];ls=q['style_lines']or q['link_lines'];sl=[dict(bbox=l['bbox'],text=l['text'],source_index=l['source_index'],boundaries=list(range(1,len(l['text']),3)),suppressed_boundaries=list(range(1,len(l['text']),2)))for l in ls]
        add('apply/'+c['label'],dict(blocks=q['blocks'],spacing_lines=sl,page_size=q['page_size']),'apply')
    for content,ls in [('international',['inter-','national']),('international international',['inter-','national']),('a \\(x\\) b',['ab']),('中文',['中文']),('中 文',['中文']),('ﬁx',['fix']),('a\u200bb',['ab']),('a\u00a0b',['ab'])]:
        for kind in ['text','code','equation','caption','doc_title','table']:
            add('projection/'+repr(content)+'/'+kind,dict(blocks=[dict(type=kind,bbox=[0.,0.,1.,1.],content=content)],page_size=[400.,600.],spacing_lines=[dict(bbox=[10.,20.,200.,30.],text=t,source_index=i,boundaries=[-1,0,1,2,3,999],suppressed_boundaries=[1,2,3])for i,t in enumerate(ls)]),'apply')
    write_json(target,dict(cases=cases));print(len(cases),'complete spacing cases;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
