"""Freeze original graphic-tail AST and full old occupied-grid proof without substitutions."""
import ast,copy,json,random,sys,math
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    edges='--edges'in sys.argv
    target=ROOT/('build/migration/native-graphic-arbitration-edges.json'if edges else'build/migration/native-graphic-arbitration.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _native_multicolumn_grid_bboxes as grids
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_AxisLine
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_detect_strong_graphic_bboxes')
    start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)and n.target.id=='output')
    wrapper=ast.parse('def _tail(source,candidates,median_height):\n pass').body[0];wrapper.body=copy.deepcopy(fn.body[start:]);module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));namespace=dict(old.__dict__);exec(compile(module,str(path),'exec'),namespace);tail=namespace['_tail']
    def source():
        rules=[dict(bbox=[20.,y,220.,y],width=1.,orientation='horizontal')for y in [50.,80.,110.,140.]]+[dict(bbox=[x,50.,x,140.],width=1.,orientation='vertical')for x in [20.,120.,220.]]
        lines=[dict(text='cell',bbox=[x,y,x+30.,y+10.],angle=0,source_index=i*2+j,visual_row_id=i)for i,y in enumerate([60.,90.,120.])for j,x in enumerate([30.,140.])]
        return dict(page_size=[400.,600.],drawing_lines=rules,lines=lines,form_bboxes=[])
    cases=[]
    def add(label,s,candidates,em=10.):
        q=clean(dict(source=copy.deepcopy(s),candidate_bboxes=copy.deepcopy(candidates),median_height=em))
        old=SimpleNamespace(**dict(s,lines=[_LineItem(**l)for l in copy.deepcopy(s.get('lines',[]))],drawing_lines=[_AxisLine(tuple(r['bbox']),r.get('width',1.),r.get('orientation','horizontal'))for r in s.get('drawing_lines',[])],form_bboxes=[tuple(b)for b in s.get('form_bboxes',[])]))
        candidates=[tuple(b)for b in candidates];ordered=sorted(candidates,key=namespace['_bbox_area'],reverse=True);rejected=[]
        for i,b in enumerate(ordered):
            proof=grids(old,em,b)
            if proof:rejected.append(dict(sorted_candidate_index=i,grid_bboxes=proof))
        expected=clean(dict(graphic_bboxes=tail(old,candidates,em),grid_rejections=rejected));cases.append(dict(label=label,request=q,expected=expected))
    big=[20.,50.,220.,140.];add('table-rejected',source(),[big]);add('empty',dict(page_size=[400.,600.],lines=[],drawing_lines=[],form_bboxes=[]),[])
    for region in [[20.,50.,220.,140.],[20.,50.,119.,140.],[20.,50.,221.,80.],[20.,50.,220.,80.],[20.,60.,220.,90.],[10.,40.,230.,150.],[0.,0.,400.,600.],[20.,50.,220.,110.],[20.,50.,220.,79.99]]:add('region-'+str(region),source(),[region])
    for count in [0,1,2,3,4]:
        s=source();s['drawing_lines']=s['drawing_lines'][:count]+s['drawing_lines'][4:];add('h-count-'+str(count),s,[big])
    for col in [0,1,2]:
        s=source();s['lines']=[l for l in s['lines']if col==2 or(l['bbox'][0]<120)==bool(col)];add('columns-'+str(col),s,[big])
    for count in [1,2,3]:
        s=source();s['lines']=s['lines'][:count*2];add('occupied-bands-'+str(count),s,[big])
    for form in [[20.,50.,220.,140.],[20.,50.,100.,140.],[300.,300.,350.,350.]]:
        s=source();s['form_bboxes']=[form];add('form-'+str(form),s,[big])
    for gap in [8.9999,9.,9.0001,10.]:
        s=source()
        for r in s['drawing_lines'][4:]:r['bbox'][3]-=gap
        add('track-coverage-'+str(gap),s,[big])
    for split in [False,True]:
        s=source()
        if split:
            s['drawing_lines']=s['drawing_lines'][:4]+[dict(bbox=[x,a,x,b],orientation='vertical',width=1.)for x in [20.,120.,220.]for a,b in [(50.,95.),(95.,140.)]]
        add('split-tracks-'+str(split),s,[big])
    for delta in [.9999,1.,1.0001,2.4999,2.5,2.5001]:
        s=source();s['drawing_lines'].append(dict(bbox=[120.+delta,50.,120.+delta,140.],orientation='vertical',width=1.));add('duplicate-track-'+str(delta),s,[big])
    for row_id in [None,0,-1]:
        s=source()
        for l in s['lines']:l['visual_row_id']=row_id
        add('row-id-'+str(row_id),s,[big])
    for angle in [0,90,180,270]:
        s=source()
        for l in s['lines']:l['angle']=angle
        add('angle-'+str(angle),s,[big])
    for delta in [4.9999,5.,5.0001]:
        s=source();s['lines'][1]['visual_row_id']=None;s['lines'][1]['bbox'][1]+=delta;s['lines'][1]['bbox'][3]+=delta;add('row-geometry-'+str(delta),s,[big])
    for em in [0.,1.,5.,10.,20.,20.0001,30.]:add('em-'+str(em),source(),[big],em)
    no=dict(page_size=[400.,600.],lines=[],drawing_lines=[],form_bboxes=[])
    for candidates in [[[0.,0.,100.,100.],[10.,10.,90.,90.]],[[0.,0.,100.,100.],[25.,0.,125.,100.]],[[0.,0.,100.,100.],[25.0001,0.,125.0001,100.]],[[0.,0.,100.,100.],[100.,0.,200.,100.]],[[0,0,100,100],[0.,0.,100.,100.]],[[0.,0.,100.,100.],[20.,0.,120.,100.],[40.,0.,140.,100.]],[[50.,100.,100.,150.],[0.,0.,20.,20.],[50.,0.,60.,10.]]]:add('merge-'+str(len(cases)),no,candidates)
    # Two independent grids must collectively cover >=70% of a spanning graphic.
    for shift in [200.,230.,300.]:
        s=source();other=source()
        for r in other['drawing_lines']:r['bbox']=[r['bbox'][0]+shift,r['bbox'][1],r['bbox'][2]+shift,r['bbox'][3]]
        for l in other['lines']:l['bbox']=[l['bbox'][0]+shift,l['bbox'][1],l['bbox'][2]+shift,l['bbox'][3]]
        s['drawing_lines']+=other['drawing_lines'];s['lines']+=other['lines'];s['page_size']=[800.,600.];add('multi-grid-'+str(shift),s,[[20.,50.,220.+shift,140.]])
    rng=random.Random(261104)
    for i in range(100):
        s=source();r=rng.choice(s['drawing_lines']);r['bbox'][rng.randrange(4)]+=rng.choice([-.01,.01,-3.,3.]);add('geometry-'+str(i),s,[big,[30.,60.,110.,75.],[50.,70.,200.,130.]])
    if edges:
        cases=[]
        a=[0,0,799999999,800000002];b=[0,0,800000000,800000001]
        large=dict(page_size=[1e9,1e9],lines=[],drawing_lines=[],form_bboxes=[])
        for order in [[a,b],[b,a],[a,list(map(float,b))],[list(map(float,b)),a]]:add('exact-area-'+str(len(cases)),large,order)
        for delta in [math.nextafter(1.,0.),1.,math.nextafter(1.,math.inf),2.49999999999999,2.5,2.50000000000001]:
            s=source();s['drawing_lines'] +=[dict(bbox=[120.+delta,50.,120.+delta,140.],width=1.,orientation='vertical')];add('track-rounding-'+str(delta),s,[big])
        for offset in [1e-12,.1,.3,1e6]:
            s=source()
            for l in s['lines']:l['bbox']=[x+offset if i%2 else x for i,x in enumerate(l['bbox'])]
            add('row-mean-'+str(offset),s,[big])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},tail_ast=ast.dump(wrapper,include_attributes=False),cases=cases));print('Frozen',len(cases),'old arbitration cases;',sum(bool(c['expected']['grid_rejections'])for c in cases),'grid rejection positive')
if __name__=='__main__':main()
