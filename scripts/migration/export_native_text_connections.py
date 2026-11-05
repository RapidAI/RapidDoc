"""Full existing Python semantic/body row connections, not simplified geometry rules."""
import copy,json,random
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_title_layout import line
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_layout as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_to_upright,_transform_axis_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TextLane,_AxisLine
    ls=lines(q['lines']);p=tuple(q['page_size']);g=[(l,_rotate_bbox_to_upright(l.bbox,p,q['angle']))for l in ls if l.angle==q['angle']];lane=_TextLane(left=q['left'],right=q['right'],lines=g,is_span=q['is_span']);tables=[tuple(b)for b in q['table_bboxes']];rules=_transform_axis_lines([_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in q['drawing_lines']],p,q['angle']);out=[]
    for i,j in q['pairs']:
        a,b=g[i],g[j];out.append(dict(semantic=old._should_connect_semantic_rows(a,b,lane,q['regular_gap'],tables,rules),text=old._should_connect_text_rows(a,b,lane,q['regular_gap'],q['gap_mad'],tables,rules),gap=old._effective_text_row_gap(a,b),body_gap=old._effective_body_text_row_gap(a,b),rule_barrier=old._horizontal_rule_separates_rows(a[1],b[1],lane,rules),table_barrier=old._connection_crosses_table(a[0].bbox,b[0].bbox,tables)))
    return out
def main():
    target=ROOT/'build/migration/native-text-connections.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('angle',0);q.setdefault('left',30.);q.setdefault('right',370.);q.setdefault('is_span',False);q.setdefault('regular_gap',4.);q.setdefault('gap_mad',1.);q.setdefault('table_bboxes',[]);q.setdefault('drawing_lines',[]);q['lines']=[project(l)for l in lines(q['lines'])];n=sum(l['angle']==q['angle']for l in q['lines']);q.setdefault('pairs',[[i,i+1]for i in range(n-1)]);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for name in ['native-final-text-lines','native-final-text-lines-chain-canonical']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if 'expected'not in c:continue
            ps=c['expected'].get('text_pages',[c['expected']]);raw=c['request'].get('pages',[c['request'].get('page')])
            for k,p in enumerate(ps):
                ls=p['remaining_lines'];source=raw[k]if raw[k]else {};source=source.get('source',source);add(c['label']+'/'+str(k),dict(lines=ls,page_size=source.get('page_size',[400.,600.])))
    def base():return dict(lines=[line('Original ordinary paragraph words continue',[30.,180.,370.,190.],0),line('Ordinary next paragraph words',[30.,194.,370.,204.],1)])
    for mode in ['base','title','title-band','title-distinct-band','doc-title','font-conflict','weight-conflict','emphasis-tail','italic-tail','hyphen','soft-hyphen','en-dash','minus','indent','repeated-indent','matrix','mixed-cjk','same-row','null-row-split','baseline','deep-overlap','rule','table','short-tail','empty-first','restored','rotated','span']:
        q=base();a,b=q['lines']
        if mode.startswith('title')or mode=='doc-title':
            for l in q['lines']:l['semantic_type']='doc_title'if mode=='doc-title'else'paragraph_title'
        if mode in ['title-band','title-distinct-band']:
            for i,l in enumerate(q['lines']):l.update(title_band_id=i if mode=='title-distinct-band'else 7,structural_title=True,explicit_section_title=True)
        if mode in ['font-conflict','weight-conflict','emphasis-tail','italic-tail']:
            a['font_signature']=['Source-Regular',0];b['font_signature']=['Source-Bold',0];b['dominant_font_weight']=700.
        if mode=='font-conflict':b['font_signature']=['Different',64]
        if mode=='weight-conflict':b['font_signature']=['Body',0]
        if mode=='emphasis-tail':a['text']+='，';b.update(text='short tail.',paragraph_terminal=True,bbox=[30.,194.,160.,204.])
        if mode=='italic-tail':b.update(text='short italic tail.',paragraph_terminal=True,bbox=[30.,194.,160.,204.],font_signature=['Source-Italic',64],dominant_font_weight=400.)
        if mode in ['hyphen','soft-hyphen','en-dash','minus']:a['text']='cross'+{'hyphen':'-','soft-hyphen':'\u00ad','en-dash':'–','minus':'−'}[mode];b.update(bbox=[80.,200.,200.,210.],font_signature=['Different',64])
        if mode in ['indent','repeated-indent','matrix']:a['bbox'][0]=60.;b['bbox'][0]=60.
        if mode=='indent':a['bbox'][0]=30.;a['text']='Original short sentence.';a['bbox'][2]=130.
        if mode=='matrix':
            for l in q['lines']:l.update(style_scale_repaired=True,source_bbox=[30.,100.,370.,200.])
        if mode=='mixed-cjk':a.update(native_cjk_style=['Chinese',0,400.,10.],effective_height=14.);b.update(native_cjk_style=['Chinese',0,400.,10.])
        if mode=='same-row':
            for l in q['lines']:l.update(visual_row_id=8,split_from_row=True)
        if mode=='null-row-split':b['split_from_row']=True
        if mode=='baseline':a['baseline']=189.;b['baseline']=201.
        if mode=='deep-overlap':b['bbox']=[30.,182.,370.,192.]
        if mode=='rule':q['drawing_lines']=[dict(bbox=[30.,192.,370.,192.],orientation='horizontal',width=.5)]
        if mode=='table':q['table_bboxes']=[[30.,191.,370.,193.]]
        if mode=='short-tail':b['bbox']=[30.,194.,100.,204.]
        if mode=='empty-first':a['text']=''
        if mode=='restored':a.update(restored_inline_cluster=True,bbox=[30.,180.,370.,205.])
        if mode=='span':q['is_span']=True
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            q.update(page_size=[600.,400.],angle=90)
            for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        add(mode,q)
    rng=random.Random(711)
    for i in range(100):
        q=base();a,b=q['lines'];a['text']=rng.choice(['Original sentence.','Original continuing words','cross-','cross–']);b.update(text=rng.choice(['Ordinary sentence.','short tail.','+ continuation.']),bbox=[rng.choice([30.,45.,60.]),rng.choice([192.,194.,198.,204.]),rng.choice([120.,250.,370.]),214.],effective_height=rng.choice([8.,10.,12.,14.]),font_coverage=rng.choice([.5,.75,1.]),font_signature=rng.choice([None,['Body',0],['Other',64]]));add('variation/'+str(i),q)
    write_json(target,dict(cases=cases));print(len(cases),'complete existing Python text/semantic row-connection cases;',sum(len(c.get('expected',[]))for c in cases),'pairs;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
