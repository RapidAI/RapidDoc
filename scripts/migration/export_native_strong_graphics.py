"""Freeze actual old strong graphic orchestration, including observed line mutations."""
import copy,json
from common import ROOT,digest,write_json
from export_native_table_page_policies import source,empty
from export_native_bar_graphics import MUTABLE,snapshot
from export_native_inline_clusters import clean

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_strong_graphic_bboxes
    raw=q['source'];projection=copy.deepcopy(raw)
    for p in projection.get('path_infos',[]):
        p.setdefault('rectangle_bboxes',[])
        if p.get('fill_rgba')is not None:p['fill_rgba']=tuple(p['fill_rgba'])
    s=source(projection);s.compound_baseline_cache={};s.isolated_path_cache={};s.drawing_component_cache=[];before=[tuple(getattr(l,k)for k in MUTABLE)for l in s.lines];boxes=_detect_strong_graphic_bboxes(s)
    result=copy.deepcopy(raw)
    if 'lines'in result:result['lines']=snapshot(raw['lines'],s.lines,before)
    return clean(dict(source=result,graphic_bboxes=boxes))
def normalized(q):
    raw=copy.deepcopy(q.get('source',{}));base=empty();base.update(raw);return dict(source=base)
def main():
    target=ROOT/'build/migration/native-strong-graphics.json';pdf_target=ROOT/'build/migration/pdf-strong-graphics.json'
    if target.exists()or pdf_target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    def add(label,q):
        q=clean(copy.deepcopy(q))
        try:out=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=out))
    add('empty',dict(source=empty()));add('empty-minimal',dict(source=dict(page_size=[400,600],lines=[],path_infos=[],image_bboxes=[])))
    corpora=['native-diagram-components','native-diagram-component-specials','native-bar-graphics','native-stacked-graphics','native-raster-charts','native-raster-charts-edges','native-concentric-graphics','native-isolated-graphics','native-raster-diagrams']
    for name in corpora:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'));all=f['cases'];step=max(1,len(all)//45)
        for i,c in enumerate(all):
            if i%step and i>=8:continue
            add(name+'/'+c['label'],normalized(c['request']))
        for c in f.get('errors',[])[:4]:add(name+'/error/'+c['label'],normalized(c['request']))
    # Path and drawing component references use source-less projections.
    for name in ['native-path-graphics','native-drawing-graphics']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases'][::max(1,len(f['cases'])//40)]:
            q=c['request'];s=empty();s.update({k:q[k]for k in ['page_size','path_infos','drawing_lines']if k in q});add(name+'/'+c['label'],dict(source=s))
    # Deliberate early gate retains invalid unconsumed source, matching old behavior.
    add('empty-unconsumed-lines',dict(source=dict(page_size=[400,600],lines=[dict(text='x',bbox=['not-read'])],path_infos=[],image_bboxes=[])))
    from export_native_diagram_components import base as diagram
    for factor in [1,2,3]:
        q=diagram();s=q['source'];s['lines']=[dict(text='body',bbox=[300.,300.,330.,330.],angle=0,effective_height=30.,source_index=0,chars=[dict(tight_bbox=[300.,300.,330.,330.])])];s['path_infos']*=factor;add('CJK-calibrated-duplicates-'+str(factor),dict(source=s))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['graphics.py','line_layout.py','isolated_graphics.py','table_detection.py','table_filled_grid.py','table_rules.py','geometry.py']];meta=dict(sources={p:digest((ROOT/p).read_bytes())for p in refs});write_json(target,dict(meta,scope='Actual old _detect_strong_graphic_bboxes and source line semantic/title/paragraph side effects. Inputs are producer source projections; no candidate verdict injection.',cases=cases,errors=errors))
    pdf=[]
    for c in json.loads((ROOT/'build/migration/pdf-complete-table-candidates.json').read_text(encoding='utf-8'))['cases']:
        q=dict(source=c['request']['source']);pdf.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],request=q,expected=reference(q)))
    write_json(pdf_target,dict(meta,scope='55 actual PDF sources through full strong graphic orchestration; no inserted candidates or altered glyph metrics.',cases=pdf));print('Frozen strong graphics',len(cases),'errors',len(errors),'source positives',sum(bool(c['expected']['graphic_bboxes'])for c in cases),'source mutations',sum(c['request']['source']!=c['expected']['source']for c in cases),'PDF positives',sum(bool(c['expected']['graphic_bboxes'])for c in pdf))
if __name__=='__main__':main()
