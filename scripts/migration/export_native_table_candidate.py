"""冻结实际旧candidate.py网格验证、索引/落格、拒绝诊断、候选评分和普通HTML。"""
import copy,json,math,random,sys
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
def main():
    extra='--extra'in sys.argv
    target=ROOT/('build/migration/native-table-candidate-extra.json'if extra else'build/migration/native-table-candidate.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import candidate as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    cases=[];html=[];rng=random.Random(6392)
    def text_from(v):
        return NativeTableText(tuple(NativeTableGlyph(**g)for g in v['glyphs']),tuple(NativeTableTextRow(**(r|dict(tokens=tuple(NativeTableToken(**t)for t in r['tokens']))))for r in v['rows']),v['median_glyph_width'],v['median_glyph_height'])
    def text(gs,tokens=()):
        glyphs=tuple(NativeTableGlyph(i,i,s,tuple(b),row)for i,(s,b,row)in enumerate(gs))
        grouped={}
        for g in glyphs:grouped.setdefault(g.visual_row,[]).append(g.glyph_id)
        rows=tuple(NativeTableTextRow(row,(0.,0.,100.,100.),tuple(NativeTableToken('token',(0.,0.,100.,100.),tuple(ids),tuple(ids))for ids in tokens if any(i in members for i in ids)),tuple(members))for row,members in grouped.items())
        return NativeTableText(glyphs,rows,5.,10.)
    def specs(rows=2,cols=2):return tuple(old.GridCellSpec(r,c,1,1,(c*50.,r*50.,(c+1)*50.,(r+1)*50.))for r in range(rows)for c in range(cols))
    def add(sp,txt,rows=2,cols=2,**options):
        request=dict(source='vector_grid',rows=rows,cols=cols,specs=[asdict(s)for s in sp],text=asdict(txt),structure_support=1.,row_stability=1.,column_stability=1.,issues=['evidence=line_grid'],allow_single_row=False,allow_single_column=False,require_atomic_tokens=False,use_grid_index=False)|options
        valid=old._validate_grid_specs(rows,cols,sp,allow_single_row=request['allow_single_row'],allow_single_column=request['allow_single_column'])
        index=old._build_grid_spec_index(rows,cols,sp)if valid and request['use_grid_index']else None
        assigned=[list(old._choose_cell_for_glyph_indexed(g,sp,index)if index else old._choose_cell_for_glyph(g,sp))for g in txt.glyphs]if valid else None
        diagnostics={};kwargs=dict(request);kwargs.update(specs=sp,text=txt,issues=tuple(request['issues']),diagnostics=diagnostics)
        value=old.build_candidate(**kwargs)
        cases.append(dict(request=request,valid=valid,index=None if index is None else asdict(index),assignments=assigned,expected=dict(candidate=None if value is None else asdict(value),diagnostics=diagnostics,html=None if value is None else old.serialize_candidate_html(value))))
    base=text([('A',[2.,2.,15.,12.],0),('B',[52.,2.,65.,12.],0),('C',[2.,52.,15.,62.],1),('D',[52.,52.,65.,62.],1)])
    for indexed in [False,True]:
        add(specs(),base,use_grid_index=indexed)
        for n,m in [(0,2),(1,1),(1,2),(2,1),(2,2),(3,3)]:
            for sr in [False,True]:
                for sc in [False,True]:add(specs(n,m),base,rows=n,cols=m,allow_single_row=sr,allow_single_column=sc,use_grid_index=indexed)
        for sp in [(),specs()[:-1],specs()+(specs()[0],),tuple(reversed(specs())),(old.GridCellSpec(-1,0,1,1,(0.,0.,20.,20.)),),tuple(old.GridCellSpec(s.row,s.col,s.rowspan,s.colspan,(s.bbox[2],s.bbox[1],s.bbox[0],s.bbox[3]))for s in specs())]:add(sp,base,use_grid_index=indexed)
        merged=(old.GridCellSpec(0,0,1,2,(0.,0.,100.,50.)),*specs()[2:]);add(merged,base,use_grid_index=indexed)
        for x in [49.,math.nextafter(49.,0),math.nextafter(49.,100),50.,51.]:
            add(specs(),text([('X',[x-10.,10.,x+10.,20.],0)]),use_grid_index=indexed)
        for total,missing in [(49,1),(50,1),(51,1),(100,2),(100,3)]:
            gs=[('x',[2.,2.,12.,12.],0)]*(total-missing)+[('?',[-20.,-20.,-10.,-10.],0)]*missing
            add(specs(),text(gs),use_grid_index=indexed)
        for total,amb in [(49,1),(50,1),(51,1),(100,2),(100,3)]:
            gs=[('x',[2.,2.,12.,12.],0)]*(total-amb)+[('?', [40.,2.,60.,12.],0)]*amb
            add(specs(),text(gs),use_grid_index=indexed)
        for atomic in [False,True]:add(specs(),text([('a',[2.,2.,10.,12.],0),('b',[52.,2.,60.,12.],0)],[(0,1)]),require_atomic_tokens=atomic,use_grid_index=indexed)
        # 物理框与逻辑列相反，捕获完整仍须让次序得分降低。
        swapped=tuple(old.GridCellSpec(s.row,1-s.col,1,1,s.bbox)for s in specs());add(swapped,base,use_grid_index=indexed)
    for value in [-1.,0.,.95,1.,2.]:
        for key in ['structure_support','row_stability','column_stability']:add(specs(),base,**{key:value})
    for _ in range(200):
        rows,cols=rng.randrange(2,6),rng.randrange(2,6);sp=list(specs(rows,cols));rng.shuffle(sp)
        gs=[]
        for i in range(rng.randrange(1,40)):
            x,y=rng.uniform(-10,cols*50),rng.uniform(-10,rows*50);gs.append((rng.choice(['A','中','&<>"\'','co-','http://x.']),[x,y,x+8.,y+10.],int(y//50)))
        add(tuple(sp),text(gs),rows=rows,cols=cols,use_grid_index=rng.choice([False,True]),require_atomic_tokens=rng.choice([False,True]),source=rng.choice(['vector_grid','text_grid','sparse_hybrid']))
    prior=json.loads((ROOT/'build/migration/native-table-text.json').read_text(encoding='utf-8'))
    for entry in prior['real']:
        c=prior['cases'][entry['case_index']]
        if c['expected']is not None:add(specs(4,5),text_from(c['expected']),rows=4,cols=5,use_grid_index=True)
    # 普通HTML保持quote=False，跨行/列属性顺序和空行均按旧版。
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableCell
    for value in ['', '<&>"\'', 'line\nbreak','中 文']:
        cs=[NativeTableCell(0,1,1,1,(0.,0.,1.,1.),value),NativeTableCell(0,0,2,2,(0.,0.,1.,1.),value)]
        html.append(dict(rows=3,cells=[asdict(c)for c in cs],expected=old.serialize_native_table_html(3,tuple(cs))))
    if extra:
        cases.clear();html.clear()
        for indexed in [False,True]:
            add(specs(),text([]),use_grid_index=indexed)
            for point in [[0.,0.,0.,0.],[50.,10.,50.,10.],[-1.,10.,-1.,10.],[49.,10.,49.,20.]]:add(specs(),text([('z',point,0)]),use_grid_index=indexed)
            wide=(old.GridCellSpec(0,0,1,2,(0.,0.,100.,50.)),old.GridCellSpec(1,0,1,2,(0.,50.,100.,100.)))
            add(wide,base,use_grid_index=indexed)
            dup=text([('x',[2.,2.,12.,12.],0)]*100)
            dup=replace(dup,glyphs=dup.glyphs[:-1]+(replace(dup.glyphs[-1],glyph_id=98,source_index=300,text='D'),))
            add(specs(),dup,use_grid_index=indexed)
            ints=tuple(replace(s,bbox=tuple(int(x)for x in s.bbox))for s in specs());add(ints,base,use_grid_index=indexed)
            for value in [1e-9,math.nextafter(1e-9,0),1e-8]:
                sp=[]
                for row,at in enumerate([-1e8,1e8,value]):sp.extend([old.GridCellSpec(row,0,1,1,(-5e8,row*50.,at,(row+1)*50.)),old.GridCellSpec(row,1,1,1,(at,row*50.,5e8,(row+1)*50.))])
                add(tuple(sp),base,rows=3,use_grid_index=indexed)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/candidate.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/text.py']
    write_json(target,dict(cases=cases,html=html,sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual old candidate builder with supplied specs/native text, including validation, indexed/brute ownership, rejection diagnostics, cell content/provenance, order score and plain HTML; does not generate physical specs or apply superscript/materialization.'))
    print(f'Frozen {len(cases)} actual old native candidates and {len(html)} plain HTML cases')
if __name__=='__main__':main()
