"""Positive original block-refinement paths, including sparse set-order/identity edges."""
import copy,json
from common import ROOT,write_json
from export_native_inline_clusters import clean
from export_native_text_blocks import STEPS
def block(t,b,rows=None,font='Body',**kw):
    b=list(map(float,b));rows=[list(map(float,r))for r in(rows or[b])]
    out=dict(type='text',content=t,bbox=b,angle=0,_text_lines=[],_reference_group=None,_visual_row_ids=set(),_single_run_row_id=None,_local_line_bboxes=rows,_local_output_line_bboxes=rows,_output_bbox_repaired=False,_line_heights=[10.]*len(rows),_font_signatures={(font,0)},_inline_math_regions=[],_paragraph_formula_context=False,_lane_interval=(30.,370.),_lane_is_span=False,_hard_break_before=False,_protected_hard_break_before=False,_explicit_break_before=False,_rule_break_before=False,_hanging_indent_group=None,_leading_emphasis_start=False)
    out.update(kw);return out
def main():
    target=ROOT/'build/migration/native-text-block-edges.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import assembly as old
    cases=[];directs=[]
    directs.append(('prefix',[block('(1',[30,180,42,190]),block('Original body words',[46,180,300,190])]))
    directs.append(('prefix',[block('•',[30,180,35,190],_protected_hard_break_before=True),block('Original body words',[60,180,300,190])]))
    directs.append(('overlap',[block('Original ordinary paragraph',[30,180,370,200],rows=[[30,180,370,190],[30,190,370,200]]),block('Original mis-split fragment',[30,190,370,200])]))
    directs.append(('overlap',[block('Original upper text',[30,160,200,200],rows=[[30,160,200,170],[30,174,200,184],[30,190,200,200]]),block('Original lower text',[205,190,370,228],rows=[[205,190,370,200],[205,204,370,214],[205,218,370,228]])]))
    directs.append(('spatial',[block('Original opener',[30,180,110,190]),block('Original body words',[30,194,370,204]),block('Original body continuation',[30,208,370,218])]))
    directs.append(('list',[block('Original introduction',[30,180,370,190]),block('(1) Original item',[30,194,370,204])]))
    directs.append(('list',[block('Original introduction',[30,170,370,180]),block('Original short tail:',[30,184,100,194]),block('(i) Original list',[30,198,370,208],_hard_break_before=True)]))
    directs.append(('unterminated',[block('Original continuing paragraph',[30,180,370,204],rows=[[30,180,370,190],[30,194,370,204]]),block('original continuation words',[30,208,370,218])]))
    directs.append(('unterminated',[block('Original sentence.',[30,180,370,190],_lane_is_span=True),block('Original following line.',[30,194,370,204],_lane_is_span=True)]))
    directs.append(('unterminated',[block('Original continuing paragraph:',[30,180,370,190]),block('(i) tail',[30,194,100,204])]))
    directs.append(('math',[block('Original wide host',[30,180,300,190]),block('x',[90,174,100,180],font='Math'),block('y',[130,190,140,196],font='Math')]))
    directs.append(('math',[block('Original wide host',[30,180,300,190]),block('x',[90,181,100,189],font='Math')]))
    directs.append(('math',[block('a b',[30,180,80,190]),block('c d',[90,180,140,190],font='Math'),block('e f',[30,194,80,204],font='Math'),block('g h',[90,194,140,204])]))
    directs.append(('math',[block('Original restored body',[30,160,370,170],_recovered_inline_math_fragments=True),block('Original intermediate body',[30,174,370,184]),block('Original restored continuation',[30,188,370,198],_recovered_inline_math_fragments=True)]))
    directs.append(('context',[block('Original formula context',[28,180,372,190],_paragraph_formula_context=True),block('Original ordinary body',[30,194,370,204]),block('Original ordinary continuation',[30,208,370,218])]))
    for n,(kind,base)in enumerate(directs):
        for mode in ['base','reverse','hard','protected','fonts','lane','span','sparse','rule-marker','formula-end']:
            bs=copy.deepcopy(base)
            if mode=='reverse':bs.reverse()
            if mode=='hard':bs[-1]['_hard_break_before']=True
            if mode=='protected':bs[-1]['_protected_hard_break_before']=True
            if mode=='fonts':bs[-1]['_font_signatures']={('Different',64)}
            if mode=='lane':bs[-1]['_lane_interval']=(100.,370.)
            if mode=='span':bs[-1]['_lane_is_span']=True
            if mode=='sparse':bs=[block('Excluded heading',[10,10,50,20],type='paragraph_title')if i%2 else block('Excluded unrelated text',[200,400+i,220,410+i])for i in range(32)]+bs
            if mode=='rule-marker':bs[-1]['_rule_break_before']=True;bs[-1]['_protected_hard_break_before']=True
            if mode=='formula-end':bs[0]['content']+='.'
            q=clean(dict(blocks=bs,page_size=[400.,600.],kind=kind));fn=getattr(old,STEPS[kind]);out=fn(bs)if kind in['list','unterminated']else fn(bs,(400.,600.));cases.append(dict(label=f'{kind}/{n}/{mode}',request=q,expected=clean(out)))
    write_json(target,dict(cases=cases));print(len(cases),'directed block edges;',sum(c['request']['blocks']!=c['expected']for c in cases),'positive changes')
if __name__=='__main__':main()
