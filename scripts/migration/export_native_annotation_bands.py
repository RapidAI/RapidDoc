"""Freeze complete existing separated-caption and symbol-legend recovery."""
import copy,json
from types import SimpleNamespace
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_inline_clusters import clean
from export_native_text_block_edges import block
from export_native_title_layout import line
from export_native_final_text_lines import typed_lines
from export_native_late_image_text import project
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.annotation_bands import recover_image_annotation_bands
    bs=typed(q['blocks'])
    recover_image_annotation_bands(bs,tuple(q['page_size']),[SimpleNamespace(**r)for r in q['drawing_lines']])
    ids={id(b):i for i,b in enumerate(bs)}
    for b in bs:
        if '_annotation_band_parent'in b:b['_annotation_band_parent_index']=ids[id(b.pop('_annotation_band_parent'))]
    return clean(bs)
def main():
    target=ROOT/'build/migration/native-annotation-bands.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,bs,rules=[]):
        q=clean(dict(blocks=bs,drawing_lines=rules,page_size=[400.,600.]));c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except(ValueError,TypeError,KeyError,AttributeError,IndexError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    def text(t,b,i,**kw):return block(t,b,_text_lines=[project(l)for l in typed_lines([line(t,b,i)])],**kw)
    for leading in [False,True]:
        image=dict(type='image',content='',bbox=[30.,80.,370.,180.],angle=0)
        bs=[image,text('Figure 1.',[30.,64.,80.,74.]if leading else[30.,184.,80.,194.],0),text('Original caption complete physical line',[30.,184.,370.,194.]if leading else[85.,184.,370.,194.],1),text('Original continuing caption line',[30.,198.,370.,208.],2)]
        for c,x in enumerate([30.,220.]):
            for r in range(4):
                y=224.+14*r;bs.extend([text('Original name',[x,y,x+85,y+10],10+c*10+r*2),text(chr(65+r),[x+90,y,x+100,y+10],11+c*10+r*2)])
        for mode in ['base','reverse','no-rule','insufficient-symbols','heading','far','footnote','other-image','narrow','zero-height']:
            v=copy.deepcopy(bs);rules=[dict(bbox=[30.,290.,370.,290.],orientation='horizontal',width=1.)]
            if mode=='reverse':v.reverse()
            if mode=='no-rule':rules=[]
            if mode=='insufficient-symbols':
                for b in v:
                    if len(b['content'])==1:b['content']='ordinary'
            if mode=='heading':v[4]['type']='paragraph_title'
            if mode=='far':v[1]['bbox'][1]+=100;v[1]['bbox'][3]+=100
            if mode=='footnote':v[3]['content']='Source: original source'
            if mode=='other-image':v.insert(0,dict(type='image',content='',bbox=[10.,80.,20.,180.],angle=0))
            if mode=='narrow':v[2]['bbox'][2]=150.
            if mode=='zero-height':v[1]['_line_heights']=[0.]
            add(f'direct/{leading}/{mode}',v,rules)
    for c in json.loads((ROOT/'build/migration/native-visual-annotations.json').read_text(encoding='utf-8'))['cases']:
        if c['kind']=='bind':add(c['label'],c['request']['blocks'])
    write_json(target,dict(cases=cases));print(len(cases),'bands;',sum(c.get('expected')!=c['request']['blocks']for c in cases if 'expected'in c),'changed;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
