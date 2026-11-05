"""Python Unicode matching/casefold and scalar formula-content contracts."""
import copy
from common import ROOT,write_json
from export_native_visual_annotations import reference
from export_native_inline_clusters import clean
from export_native_page_block_rules import typed
from export_native_text_block_edges import block
def main():
    target=ROOT/'build/migration/native-visual-order-contracts.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _sort_blocks_with_visual_row_groups
    cases=[]
    def add(label,bs,kind):
        q=clean(dict(blocks=bs,page_size=[400.,600.]));c=dict(label=label,kind=kind,request=q)
        if kind=='bind':c['expected']=reference(q,'bind')
        else:
            bs=typed(q['blocks']);out=_sort_blocks_with_visual_row_groups(bs,(400.,600.));ids={id(b):i for i,b in enumerate(bs)};c['expected']=clean(dict(blocks=bs,order=[ids[id(b)]for b in out]))
        cases.append(c)
    for first,second in [('1İ','1i'),('1ı','1i'),('1İ','1İ'),('1ſ','1s'),('1K','1k'),('Ⅰ','I'),('A 1','A1'),('A.1','A.1'),('1a','1A')]:
        bs=[dict(type='image',bbox=[30.,80.,370.,180.],content='',angle=0),block('图'+first+' 原始标题',[30,184,370,194]),block('Figure '+second+'. Original caption',[30,208,370,218])]
        add(first+'/'+second,bs,'bind')
    for word in ['İSmith','ıSmith','ſSmith','Smith','KSmith']:
        add(word,[dict(type='image',bbox=[30.,80.,370.,180.],content='',angle=0),block('('+word+' 2020).',[30,184,370,194])],'bind')
    for content in [None,True,False,0,12,{},[],['(1)'],'(1)','… (1)','(1)\n']:
        bs=[dict(type='image',bbox=[20.,300.,50.,330.],content=content,angle=0),block('(1)',[320,180,350,190]),dict(type='equation',bbox=[100.,180.,300.,190.],content='',angle=0)]
        add('scalar/'+str(content),bs,'order')
    write_json(target,dict(cases=cases));print(len(cases),'Unicode/scalar contracts')
if __name__=='__main__':main()
