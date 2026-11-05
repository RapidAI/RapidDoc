"""Exact existing final reference-band/narrow-graphic column ordering."""
import copy,json
from types import SimpleNamespace
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_inline_clusters import clean
from export_native_title_layout import line
from export_native_final_text_lines import typed_lines
from export_native_late_image_text import project
from export_native_text_block_edges import block
def main():
    target=ROOT/'build/migration/native-final-order-edges.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import order_body_above_reference_band
    cases=[]
    def add(label,bs,p,kind,page):
        q=clean(dict(blocks=bs,page_size=p,kind=kind,page=page));bs=typed(q['blocks']);c=dict(label=label,request=q)
        try:c['expected']=clean(old._preserve_narrow_graphic_column_order(bs,tuple(p))if kind=='narrow_order'else order_body_above_reference_band(bs,SimpleNamespace(**dict(page,page_size=tuple(p)))))
        except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    def base():
        bs=[]
        for col,y in [(0,180.),(1,178.),(0,220.),(1,218.)]:
            x=30.+col*190.;boxes=[[x,y+14*i,x+150.,y+10.+14*i]for i in range(2)];ls=[project(l)for l in typed_lines([line('Original ordinary paragraph words',b,len(bs)*2+i)for i,b in enumerate(boxes)])];bs.append(block('Original ordinary paragraph words',[x,y,x+150.,y+24.],rows=boxes,_text_lines=ls,_lane_interval=(x,x+150.)))
        return bs
    for mode in ['base','reverse','marginals','graphic','barrier','wide','empty-member','ref-group','small-body','disabled','empty-regions']:
        bs=base();page=dict(numbered_references=True,reference_regions=[[30.,300.,370.,580.]])
        if mode=='reverse':bs.reverse()
        if mode=='marginals':bs.insert(1,dict(type='header',content='Original header',bbox=[30.,10.,370.,20.],angle=0))
        if mode in ['graphic','barrier']:bs.insert(1,dict(type='image',content='',bbox=[220.,120.,370.,160.],angle=0,_caption_graphic=True))
        if mode=='wide':bs.append(block('Original spanning text',[30.,270.,370.,280.]))
        if mode=='empty-member':bs[0]['_text_lines']=[]
        if mode=='ref-group':bs[0]['_reference_group']=7
        if mode=='small-body':bs[0]['bbox'][2]=bs[0]['bbox'][0]+5.
        if mode=='disabled':page['numbered_references']=False
        if mode=='empty-regions':page['reference_regions']=[]
        for kind in ['reference_order','narrow_order']:
            if kind=='narrow_order':bs[0]['_caption_graphic']=True
            add('direct/'+mode+'/'+kind,bs,[400.,600.],kind,page)
    f=json.loads((ROOT/'build/migration/native-final-refined-blocks-chain-paths.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'expected'not in c:continue
        for i,(p,s)in enumerate(zip(c['expected']['pages'],c['expected']['block_pages'])):
            bs=p.get('fixed_blocks',[])+s['formula_blocks']+s['index_blocks']+s['refined_text_blocks'];page=dict(numbered_references=p.get('numbered_references',False),reference_regions=p.get('reference_regions',[]))
            for kind in ['reference_order','narrow_order']:add(c['label']+'/'+str(i)+'/'+kind,bs,p.get('page_size',p.get('source',{}).get('page_size')),kind,page)
    write_json(target,dict(cases=cases));print(len(cases),'complete final-order guard cases;',sum(c.get('expected')!=c['request']['blocks']for c in cases if 'expected'in c),'positive changes')
if __name__=='__main__':main()
