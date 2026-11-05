"""Original superscript references, hanging notes and real lowercase ink proofs."""
import copy
from common import ROOT,write_json
from export_native_footnotes import base,reference
from export_native_geometry import lines
from export_native_late_image_text import project

def main():
    p=ROOT/'build/migration/native-early-footnote-edges.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['lines']=[project(l)for l in lines(q['lines'])];q['reference_lines']=[project(l)for l in lines(q['reference_lines'])];cases.append(dict(label=label,request=q,expected=reference(q)))
    def chars(text,y,ink=8.,marker_slot=False):
        result=[];x=30.
        for i,c in enumerate(text):
            if marker_slot and i==3:x+=12.
            result.append(dict(char=c,char_idx=i,bbox=[x,y,x+5.,y+12.],tight_bbox=[x,y+12.-ink,x+5.,y+12.],origin=[x,y+12.],font=dict(name='Body',size=12.,flags=0)));x+=5.
        return result
    for mode in ['chart','no-reference','same-size-reference','nonraised','bold-note','sequential','skip-number','font-change','unicode-reference','unicode-mismatch','spanning','spanning-no-slot']:
        q=base();q['drawing_lines']=[];q['visual_bboxes']=[[30.,300.,370.,450.]];q['lines'][-2].update(text='1 Original native chart note begins here',effective_height=12.);q['lines'][-1]['effective_height']=12.
        ref=copy.deepcopy(q['lines'][0]);ref['chars']=chars('native1',150.);ref['chars'][-1].update(char='1',bbox=[60.,150.,64.,157.],origin=[60.,157.]);q['reference_lines']=[ref]
        if mode=='no-reference':q['reference_lines']=[]
        if mode=='same-size-reference':ref['chars'][-1]['bbox'][3]=162.
        if mode=='nonraised':ref['chars'][-1]['origin'][1]=162.
        if mode=='bold-note':q['lines'][-2]['dominant_font_weight']=700.
        if mode in ['sequential','skip-number','font-change']:
            q['lines'][-1]['text']=('3'if mode=='skip-number'else'2')+' Original next chart note'
            if mode=='font-change':q['lines'][-1]['font_signature']=['Other',0]
        if mode in ['unicode-reference','unicode-mismatch']:q['lines'][-2]['text']='١ Original native chart note begins here';ref['chars'][-1]['char']='١'if mode=='unicode-reference'else'1'
        if mode.startswith('spanning'):
            q['visual_bboxes']=[];q['lines'][0]['chars']=ref['chars'];q['reference_lines']=[]
            for l in q['lines'][:5]:l['bbox'][0]=50. if mode=='spanning'else 30.
            q['lines'][-2].update(text='1. Original native spanning note begins here',bbox=[30.,500.,370.,512.]);q['lines'][-1]['bbox']=[30.,516.,370.,528.]
        add(mode,q)
    for mode in ['hanging','no-prefix','one-continuation','large-font','single-ink','same-ink','marker-gap','few-ink-body','bold-ink']:
        q=base();q['drawing_lines'][0]['bbox']=[30.,490.,370.,490.];q['lines'][-2].update(bbox=[30.,510.,250.,522.],source_bbox=[30.,510.,38.,522.],effective_height=12.,text='1. original native single note words');q['lines'][-1].update(bbox=[50.,526.,250.,538.],effective_height=12.);q['lines'].append(dict(q['lines'][-1],bbox=[50.,542.,250.,554.],source_index=12))
        if mode=='no-prefix':q['lines'][-3]['source_bbox']=None
        if mode=='one-continuation':q['lines']=q['lines'][:-1]
        if mode=='large-font':q['lines'][-3]['effective_height']=16.
        if mode in ['single-ink','same-ink','marker-gap','few-ink-body','bold-ink']:
            q['lines']=q['lines'][:-2];q['lines'][-1]['source_bbox']=None;q['lines'][-1]['chars']=chars(q['lines'][-1]['text'],510.,ink=6. if mode!='same-ink'else 10.,marker_slot=mode!='marker-gap')
            for i,l in enumerate(q['lines'][:-1]):l['chars']=chars('originalbodyletters',l['bbox'][1],ink=10.);l['chars']=l['chars'][:5]if mode=='few-ink-body'else l['chars']
            if mode=='bold-ink':q['lines'][-1]['dominant_font_weight']=700.
        add(mode,q)
    write_json(p,dict(cases=cases));print(len(cases),'superscript/hanging/ink scenes;',sum(c['request']['lines']!=c['expected']['lines']for c in cases),'mutating scenes')
if __name__=='__main__':main()
