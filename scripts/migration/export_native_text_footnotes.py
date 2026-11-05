"""Full original grouped-footnote assembly, real chain inputs plus directed markers."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.footnotes import _build_grouped_page_footnote_blocks
    b,c=_build_grouped_page_footnote_blocks(lines(q['lines']),[set(g)for g in q['page_footnote_groups']],tuple(q['page_size']));out=clean(dict(blocks=b,consumed=sorted(c)))
    for b in out['blocks']:b['_visual_row_ids'].sort()
    return out
def main():
    target=ROOT/'build/migration/native-text-footnotes.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('page_footnote_groups',[[l['source_index']for l in q['lines']]]);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for name in ['native-final-text-lines','native-final-text-lines-chain-canonical']:
        for c in json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']:
            if 'expected'not in c:continue
            for i,p in enumerate(c['expected'].get('text_pages',[c['expected']])):add(name+'/'+c['label']+'/'+str(i),dict(lines=p['remaining_lines'],page_footnote_groups=p['page_footnote_groups']))
    def make(texts,lefts=None,widths=None,ys=None):
        return [line(t,[float((lefts or[30]*len(texts))[i]),float((ys or[480+j*14 for j in range(len(texts))])[i]),float((lefts or[30]*len(texts))[i]+(widths or[340]*len(texts))[i]),float((ys or[480+j*14 for j in range(len(texts))])[i]+10)],i,semantic_type='page_footnote')for i,t in enumerate(texts)]
    base=[('empty',[]),('single',make(['Original footnote'])),('numbered',make(['1. Original footnote','Original continuation','2. Next footnote','Original continuation'],lefts=[30,50,30,50],widths=[340,300,340,300])),('institution',make(['Author affiliations','1 Original University; 2 Original Institute','Original department continuation'])),('hanging',make(['Original footnote entry','Original continuation','Next footnote entry','Next continuation'],lefts=[30,50,30,50],widths=[250,300,250,300])),('indented',make(['Original footnote entry','Original continuation','Next footnote entry','Next continuation'],lefts=[50,30,50,30],widths=[280,300,280,300])),('short-notes',make(['12','同上','13','原注'],lefts=[30,48,30,48],widths=[10,35,10,35],ys=[480,480,505,505])),('prefix',make(['Original independent prefix','1','Original first note','Original continuation','2','Original second note'],lefts=[30,30,45,45,30,45],widths=[180,5,300,300,5,300],ys=[460,480,480,494,520,520])),('superscript',make(['Original independent prefix','1Original first note','Original continuation','2Original second note'],widths=[150,330,300,330]))]
    for name,ls in base:
        if name=='short-notes':
            for i,l in enumerate(ls):l.update(visual_row_id=i//2,split_from_row=True)
        if name=='superscript':
            for l in [ls[1],ls[3]]:
                l['chars']=[dict(char=c,char_idx=i,bbox=[30.+5*i,480.,35.+5*i,486. if i==0 else 490.],origin=[30.+5*i,485. if i==0 else 490.],font=dict(name='Body',size=6. if i==0 else 10.,flags=0,weight=400.))for i,c in enumerate(l['text'])]
        for mode in ['base','reverse','no-row','marker-start','ink','rotation','duplicates','empty-content','groups','untyped']:
            q=dict(lines=copy.deepcopy(ls))
            if mode=='reverse':q['lines'].reverse()
            if mode=='no-row':
                for l in q['lines']:l.update(visual_row_id=None,split_from_row=False)
            if mode=='marker-start':
                for i,l in enumerate(q['lines']):l['footnote_marker_start']=i%2==0
            if mode=='ink':
                for l in q['lines']:l['ink_bbox']=[l['bbox'][0]+1,l['bbox'][1]+1,l['bbox'][2]-1,l['bbox'][3]-1]
            if mode=='rotation':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                q['page_size']=[600.,400.]
                for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
            if mode=='duplicates':
                for i,l in enumerate(q['lines']):l['source_index']=[-1,-2,32,7][i%4]
            if mode=='empty-content':
                for l in q['lines']:l['text']=''
            if mode=='groups':q['page_footnote_groups']=[[l['source_index']for l in q['lines']],[-999],[l['source_index']for l in q['lines'][::2]]]
            if mode=='untyped'and q['lines']:q['lines'][0]['semantic_type']=None
            add(name+'/'+mode,q)
    write_json(target,dict(cases=cases));print(len(cases),'complete grouped-footnote cases;',sum(len(c.get('expected',{}).get('blocks',[]))for c in cases),'blocks;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
