"""Original dense-row restoration and complete formula-input sequence."""
import copy,json,random
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_merging as merge,formulas
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    ls=lines(q['lines']);size=tuple(q['page_size']);tables=[tuple(b)for b in q['table_bboxes']]
    out=merge._restore_dense_split_visual_rows(ls,size,tables)
    if q.get('dense_only'):return clean(dict(lines=[project(l)for l in out],source_lines=[project(l)for l in ls]))
    out=merge._merge_same_baseline_text_lines(out,size,tables)
    rules=[_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in q['drawing_lines']]
    blocks,remaining=formulas._build_formula_like_blocks(out,tables,size,drawing_lines=rules)
    return clean(dict(formula_blocks=blocks,remaining_lines=[project(l)for l in remaining],formula_input=[project(l)for l in out]))

def main():
    target=ROOT/'build/migration/native-formula-input.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('table_bboxes',[]);q.setdefault('drawing_lines',[]);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-formula-full.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'kind'not in c['request']:add(c['label'],c['request'])
    def base(n=2):return dict(lines=[line('Original complete body paragraph',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])]+[dict(line(t,[30.+i*100.,190.,125.+i*100.,200.],20+i,visual_row_id=9,split_from_row=True,run_index=i),median_glyph_width=4.)for i,t in enumerate(['Original dense' if i==0 else 'body words'for i in range(n)])])
    for mode in ['same-lane','two-only','three','seven','preserved','font','semantic','candidate','bad-runs','no-row','different-row','table','overlap','occupancy','gap','height','ink','bullet','membership','split-title','null-font','native-chars','rotated','duplicate-source']:
        q=base(3 if mode=='three'else 7 if mode=='seven'else 2)
        if mode=='two-only':q['lines']=q['lines'][6:]
        if mode=='preserved':q['lines'][-1]['preserve_split_boundary']=True
        if mode=='font':q['lines'][-1]['font_signature']=['Other',0]
        if mode=='semantic':q['lines'][-1]['semantic_type']='paragraph_title'
        if mode=='candidate':q['lines'][-1]['formula_candidate_only']=True
        if mode=='bad-runs':q['lines'][-1]['run_index']=5
        if mode=='no-row':q['lines'][-1]['visual_row_id']=None
        if mode=='different-row':q['lines'][-1]['visual_row_id']=10
        if mode=='table':q['table_bboxes']=[[20.,185.,150.,210.]]
        if mode=='overlap':q['lines'][-1]['bbox']=[120.,190.,225.,200.]
        if mode=='occupancy':q['lines'][-1]['bbox']=[145.,190.,165.,200.]
        if mode=='gap':q['lines'][-1]['bbox']=[160.,190.,255.,200.]
        if mode=='height':q['lines'][-1].update(effective_height=15.,bbox=[130.,185.,225.,200.])
        if mode=='ink':
            for l in q['lines'][6:]:l['ink_bbox']=[l['bbox'][0]+1,l['bbox'][1]+1,l['bbox'][2]-1,l['bbox'][3]-1]
        if mode=='bullet':q['lines'][6].update(text='•',bbox=[30.,190.,38.,200.]);q['lines'][7]['bbox']=[40.,190.,225.,200.]
        if mode=='membership':
            for l in q['lines'][6:]:l.update(paragraph_group=7,title_band_id=8,structural_title=True,explicit_section_title=True,title_suppressed=True)
        if mode=='split-title':q['lines'][6].update(text='A.',native_title_label_left=40.);q['lines'][7]['native_title_label_left']=130.
        if mode=='null-font':
            for l in q['lines'][6:]:l['font_signature']=None
        if mode=='native-chars':
            for l in q['lines'][6:]:l['chars']=[dict(char=c,bbox=[l['bbox'][0]+i*4,190.,l['bbox'][0]+i*4+4,200.],char_idx=l['source_index']*100+i,font=dict(name='Body',size=10.,flags=0,weight=400.))for i,c in enumerate(l['text'])]
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            q['page_size']=[600.,400.]
            for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        if mode=='duplicate-source':q['lines'][-1]['source_index']=20
        add('dense/'+mode,dict(q,dense_only=True));add('sequence/'+mode,q)
    write_json(target,dict(cases=cases));print(len(cases),'original formula-input sequences;',sum(len(c.get('expected',{}).get('lines',[]))<len(c['request']['lines'])for c in cases if c['request'].get('dense_only')),'dense restorations')
if __name__=='__main__':main()
