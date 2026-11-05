"""Freeze original bar function through label_anchor; table and admission remain real rules."""
import ast,copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_tiled_tables import old_source
from export_native_bar_geometry import reference as admission_reference

def group_reference():
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as old
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_detect_native_bar_graphics'));fn.name='_bar_groups';fn.returns=None;fn.args=ast.parse('def f(source,em):pass').body[0].args
    outer=next(n for n in fn.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='horizontal');loop=next(n for n in outer.body if isinstance(n,ast.While));cluster=next(n for n in loop.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='cluster');stop=next(i for i,n in enumerate(cluster.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='label_anchor'for t in n.targets));cluster.body=cluster.body[:stop+1]
    zero=next(i for i,n in enumerate(cluster.body)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_bbox_union_many'and '_bar_zero_axis_bboxes'in ast.unparse(n));cluster.body.insert(zero,ast.parse('core_bbox = bounds').body[0])
    cluster.body+=ast.parse('outputs.append(dict(horizontal=horizontal,baseline=baseline,bar_indices=[next(j for j,p in enumerate(paths) if p is bars[i]) for i in cluster],numeric_line_indices=[j for j,l in enumerate(source.lines) if any(l is n for n in numeric)],core_bbox=core_bbox,bbox=bounds,label_anchor=label_anchor))').body
    fn.body[-1]=ast.parse('return outputs,table_bounds').body[0];ns=dict(old.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns);return ns['_bar_groups'],ast.dump(fn,include_attributes=False)

def reference(s,em):
    from types import SimpleNamespace
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_tiled_fill_table_candidates
    old=old_source(s);old.compound_baseline_cache={};groups,tables=group_reference()[0](old,em);admit,_,_=admission_reference();admitted=[]
    for i,p in enumerate(old.path_infos):
        paths,directions,_=admit(SimpleNamespace(path_infos=[p],drawing_lines=old.drawing_lines,compound_baseline_cache={}),em,tables)
        for b in paths:
            ri=next((j for j,box in enumerate(p.rectangle_bboxes)if box is b.bbox),None)if len(p.rectangle_bboxes)>1 else None
            admitted.append(dict(path_index=i,rectangle_index=ri,bbox=b.bbox,horizontal=any(b is x for x in directions[0]),vertical=any(b is x for x in directions[1])))
    return clean(dict(table_bboxes=tables,admitted_bars=admitted,groups=groups))

def base(horizontal=True,compound=False):
    boxes=[[50.,100.+i*20.,130.+i*20.,110.+i*20.]for i in range(4)]if horizontal else[[100.+i*20.,200.-(80.+i*20.),110.+i*20.,200.]for i in range(4)]
    def p(b,rs):return dict(bbox=b,rectangle_bboxes=rs,fill_visible=True,stroke_visible=False,fill_rgba=[0,100,200,255],segment_count=5,form_depth=0,source_index=0)
    if compound:paths=[p([min(b[0]for b in boxes),min(b[1]for b in boxes),max(b[2]for b in boxes),max(b[3]for b in boxes)],boxes)]
    else:paths=[p(b,[])for b in boxes]
    labels=[dict(text=str(i*20),bbox=[20.,102.+i*20.,40.,112.+i*20.],source_index=i,chars=[])for i in range(4)]if horizontal else[dict(text=str(i*20),bbox=[100.+i*20.,210.,110.+i*20.,220.],source_index=i,chars=[])for i in range(4)]
    rules=[dict(bbox=[50.,95.,50.,175.],orientation='vertical',width=2.)]if horizontal else[dict(bbox=[95.,200.,175.,200.],orientation='horizontal',width=2.)]
    return dict(page_size=[400.,600.],chars=[],path_infos=paths,drawing_lines=rules,lines=labels)

def main():
    target=ROOT/'build/migration/native-bar-groups.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,s,em=10.):cases.append(dict(label=label,request=clean(dict(source=copy.deepcopy(s),em=em)),expected=reference(s,em)))
    for h in [True,False]:
        for comp in [False,True]:add(f'base-{h}-{comp}',base(h,comp))
        for count in [0,1,2]:s=base(h);s['lines']=s['lines'][:count];add(f'numeric-count-{h}-{count}',s)
        for text in ['+12','−12','-12','１２','١٢','12,3','12.3%',' 12\u00a0','12 %','1e2','12.','１２.５','%12','⁴','12\n3','1_000']:
            s=base(h)
            for l in s['lines']:l['text']=text
            add(f'grammar-{h}-{text!r}',s)
        for gap in [.99999,1.,1.00001,50.,50.00001]:
            s=base(h);axis=1 if h else 0
            for i,p in enumerate(s['path_infos']):p['bbox'][axis]=100.+i*(10.+gap);p['bbox'][axis+2]=110.+i*(10.+gap)
            add(f'gap-{h}-{gap}',s)
        for delta in [3.49999,3.5,3.50001]:
            s=base(h);axis=0 if h else 3;s['path_infos'][-1]['bbox'][axis]+=delta;add(f'baseline-{h}-{delta}',s)
        s=base(h);s['path_infos']*=2;add('duplicates-'+str(h),s)
        s=base(h)
        for p in s['path_infos'][:2]:
            b=p['bbox']
            if h:b[0],b[2]=-b[2]+100.,50.
            else:b[1],b[3]=200.,400.-b[1]
        add('signed-bars-'+str(h),s)
    for scale in [0.1,1.,10.]:
        s=base()
        for p in s['path_infos']:p['bbox']=[x*scale for x in p['bbox']]
        for r in s['drawing_lines']:r['bbox']=[x*scale for x in r['bbox']];r['width']*=scale
        for l in s['lines']:l['bbox']=[x*scale for x in l['bbox']]
        add('scale-'+str(scale),s,10.*scale)
    s=base();s['path_infos']+=copy.deepcopy(s['path_infos'])
    for p in s['path_infos'][4:]:p['bbox']=[x+(200. if i%2==0 else 0.)for i,x in enumerate(p['bbox'])]
    for l in copy.deepcopy(s['lines']):l['bbox']=[x+(200. if i%2==0 else 0.)for i,x in enumerate(l['bbox'])];s['lines'].append(l)
    add('two-groups',s)
    # Consume the complete tiled-fill stage as real upstream evidence, including headers.
    from export_native_tiled_tables import base as tiled
    for header in [False,True]:
        s=tiled(header=header)
        for p in s['path_infos']:p['fill_rgba']=[0,100,200,255]
        add('actual-upstream-table-'+str(header),s)
    rng=random.Random(261110)
    for i in range(100):
        s=base(bool(i%2),bool(i%3==0));rng.shuffle(s['path_infos']);rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines'])
        if not i%3:
            s['lines'][0]['bbox']=[50.,100.,70.,110.]
        add('permuted-'+str(i),s)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_graphic_geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},prefix_ast=group_reference()[1],cases=cases));print('Frozen',len(cases),'bar group cases;',sum(bool(c['expected']['groups'])for c in cases),'group-positive;',sum(bool(c['expected']['table_bboxes'])for c in cases),'table exclusions')
if __name__=='__main__':main()
