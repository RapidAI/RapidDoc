"""Full existing Python final metric/resolved-title/post-semantic mergers."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line
KINDS=['metric','resolved','post','sequence']
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_merging as old
    ls=lines(q['lines']);p=tuple(q['page_size']);tables=[tuple(b)for b in q['table_bboxes']];kind=q['kind']
    if kind in ['metric','sequence']:ls=old._merge_metric_heading_fragments(ls,p)
    if kind in ['resolved','sequence']:ls=old._merge_title_resolved_visual_rows(ls,p)
    if kind in ['post','sequence']:ls=old._merge_post_semantic_text_runs(ls,p,tables)
    return clean(dict(lines=[project(l)for l in ls]))
def main():
    target=ROOT/'build/migration/native-semantic-merges.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('table_bboxes',[]);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for name in ['native-final-title-rules-canonical','native-page-merging','native-formula-input']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for i,c in enumerate(f['cases']):
            if i%29 or 'expected'not in c:continue
            expected=c['expected'];ls=expected if isinstance(expected,list)else expected.get('lines',c['request'].get('lines'));ls=ls if isinstance(ls,list)else c['request']['lines']
            for kind in KINDS:add(name+'/'+c['label']+'/'+kind,dict(c['request'],lines=ls,kind=kind))
    def row(t,b,i,**kw):return dict(line(t,b,i,**kw),split_from_row=True,visual_row_id=5,run_index=i)
    for mode in ['metric','resolved','numbered','post','shared','justified','caption','three','same-font','sparse-prefix','table','barrier','side-column','source-order','rotated','paired','native-chars','em-different','no-lines','one-line']:
        q=dict(lines=[row('Original',[30.,190.,100.,200.],0,font='Body'),row('text words',[103.,190.,225.,200.],1,font='Other')])
        if mode=='metric':q['lines']=[line('20%', [30.,190.,80.,210.],0,h=20.,font='Heading'),line('↑2',[85.,190.,110.,210.],1,h=20.,font='Heading',semantic_type='paragraph_title'),line('Second metric',[230.,190.,370.,210.],2,h=20.,font='Heading',semantic_type='paragraph_title')]
        if mode=='resolved':
            for l in q['lines']:l['semantic_type']='paragraph_title'
        if mode=='numbered':q['lines']=[line('2.',[30.,190.,45.,200.],0,semantic_type='paragraph_title'),line('Original heading',[50.,190.,200.,200.],1,semantic_type='paragraph_title')]
        if mode in ['shared','justified']:q['lines'][1].update(font_signature=['Body',0],bbox=[120.,190.,300.,200.]);q['lines'].append(line('Original following complete paragraph',[30.,207.,300.,217.],2))
        if mode=='caption':q['lines'][1]['caption_start']=True
        if mode=='three':q['lines'].append(row('trailing',[228.,190.,310.,200.],2))
        if mode=='same-font':q['lines'][1]['font_signature']=['Body',0]
        if mode=='sparse-prefix':q['lines'][0].update(text='A.',bbox=[30.,190.,40.,200.]);q['lines'][1].update(font_signature=['Body',0],bbox=[80.,190.,310.,200.])
        if mode=='table':q['table_bboxes']=[[99.,185.,104.,205.]]
        if mode=='barrier':q['lines'] += [line('Original left body paragraph',[30.,y,160.,y+10.],i+10)for i,y in enumerate([100.,120.,140.])]+[line('Original right body paragraph',[230.,y,370.,y+10.],i+20)for i,y in enumerate([100.,120.,140.])];q['lines'][1]['bbox']=[230.,190.,370.,200.]
        if mode=='side-column':q['lines'][1]['bbox']=[230.,190.,370.,200.]
        if mode=='source-order':q['lines'].reverse()
        if mode=='em-different':q['lines'][1].update(effective_height=14.,em_height=8.,style_scale_repaired=True)
        if mode=='paired':q['lines'][1]['bbox']=[100.,196.,120.,204.];q['lines'][1]['effective_height']=8.
        if mode=='native-chars':
            for l in q['lines']:l['chars']=[dict(char=c,char_idx=l['source_index']*100+i,bbox=[l['bbox'][0]+i*4.,190.,l['bbox'][0]+i*4.+4.,200.],font=dict(name='Body',size=10.,flags=0,weight=400.))for i,c in enumerate(l['text'])]
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            q['page_size']=[600.,400.]
            for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        if mode=='no-lines':q['lines']=[]
        if mode=='one-line':q['lines']=q['lines'][:1]
        for kind in KINDS:add(mode+'/'+kind,dict(q,kind=kind))
    write_json(target,dict(cases=cases));print(len(cases),'existing Python final merge cases;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
