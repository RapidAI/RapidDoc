"""Existing full grouped final-page ordering, preserving region identity and all tail passes."""
import copy,json
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_inline_clusters import clean
from export_native_text_block_edges import block
def main():
    target=ROOT/'build/migration/native-grouped-order.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _sort_blocks_with_visual_row_groups
    from loguru import logger
    logger.remove();cases=[]
    def add(label,bs,p=[400.,600.],regions=[]):
        q=clean(dict(blocks=bs,page_size=p,regions=regions));c=dict(label=label,request=q)
        try:
            bs=typed(q['blocks']);rs=[[bs[i]for i in ids]for ids in regions];out=_sort_blocks_with_visual_row_groups(bs,tuple(p),visual_annotation_regions=rs);ids={id(b):i for i,b in enumerate(bs)};c['expected']=clean(dict(blocks=bs,order=[ids[id(b)]for b in out]))
        except(ValueError,TypeError,IndexError,KeyError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for i,c in enumerate(json.loads((ROOT/'build/migration/native-visual-annotations.json').read_text(encoding='utf-8'))['cases']):
        if c['kind']!='bind'or'expected'not in c:continue
        e=c['expected'];add(c['label'],e['blocks'],c['request']['page_size'],e['regions'])
    for label,bs in [('inline',[block('Original first text',[30,180,180,204],rows=[[30,180,180,190],[30,194,180,204]],_visual_row_ids={7}),dict(type='image',bbox=[190.,180.,210.,190.],content='',angle=0,_inline_visual_row_id=7),block('Original right text',[220,180,370,190],_visual_row_ids={7})]),('marginals',[block('Header right',[250,10,370,20],type='header'),block('Header left',[30,11,150,21],type='header'),block('Footnote right',[220,500,370,580],type='page_footnote'),block('Footnote left',[30,500,180,580],type='page_footnote'),block('Footer',[30,590,370,600],type='footer'),block('Body',[30,180,370,200])]),('equation',[block('(1)',[320,180,350,190]),dict(type='equation',bbox=[100.,180.,300.,190.],content='',angle=0)]),('leading',[block('Original section',[30,65,150,75],type='paragraph_title'),block('Original title',[140,40,260,60],type='doc_title')]),('steps',[block('1. Original parent',[30,170,370,180]),block('a. Original child',[45,184,370,194]),block('b. Original child',[45,198,370,208]),block('c. Original child',[45,212,370,222]),block('2. Original next parent',[30,226,370,236])]),('panels',sum(([block('Original panel heading',[x,100,x+70,110],type='paragraph_title'),block('Original panel body',[x,120,x+90,170])]for x in [20,150,280]),[])),('visual-row',[block('Original left',[30,180,180,190],_single_run_row_id=7),block('Original right',[220,180,370,190],_single_run_row_id=7)])]:
        for mode in ['base','reverse','missing-row','spans','barrier','rotated']:
            v=copy.deepcopy(bs)
            if mode=='reverse':v.reverse()
            if mode=='missing-row':
                for b in v:b.pop('_single_run_row_id',None);b.pop('_inline_visual_row_id',None);b.pop('_visual_row_ids',None)
            if mode=='spans':
                for b in v:b['_lane_is_span']=True
            if mode=='barrier':v.append(dict(type='image',bbox=[5.,80.,395.,160.],content='',angle=0))
            if mode=='rotated':
                for b in v:b['angle']=90
            add('direct/'+label+'/'+mode,v)
    write_json(target,dict(cases=cases));print(len(cases),'full original grouped ordering cases;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
