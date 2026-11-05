"""Freeze actual old block preparation functions; inputs and side effects remain independent."""
import copy,json,random,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    typed='--types-only'in sys.argv
    target=ROOT/('build/migration/native-table-block-types.json'if typed else'build/migration/native-table-block.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_PageSource,_AxisLine,_TableCandidate,_TableAnnotation
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_materialization as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    cases=[];real=[]
    def annotation(v):
        return _TableAnnotation(v['kind'],tuple(v['bbox']),set(v['line_indices']),{int(k):tuple(b)for k,b in v['line_bboxes'].items()})
    def add(label,s,c,failed=(),calibrated=None):
        request=copy.deepcopy(dict(source=s,candidate=c,failed_annotations=list(failed),calibrated_char_bboxes=calibrated))
        ls=[]
        for v in copy.deepcopy(s['lines']):
            v['bbox']=tuple(v['bbox'])
            if v.get('font_signature')is not None:v['font_signature']=tuple(v['font_signature'])
            ls.append(_LineItem(**v))
        source=_PageSource(tuple(s['page_size']),ls,copy.deepcopy(s['chars']),[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in s['drawing_lines']])
        v=copy.deepcopy(c);v['bbox']=tuple(v['bbox']);v['local_bbox']=tuple(v['local_bbox']);v['core_bbox']=None if v['core_bbox']is None else tuple(v['core_bbox']);v['line_indices']=set(v['line_indices']);v['annotations']=[annotation(a)for a in v['annotations']];v['inferred_grid']=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in v['inferred_grid']]
        candidate=_TableCandidate(**v)
        repairs=old._overlapping_table_character_repairs(source)|({int(k):tuple(b)for k,b in calibrated.items()}if calibrated else {})
        native=[dict(q,bbox=repairs[q['char_idx']])if q.get('char_idx')in repairs else q for q in source.chars]
        old._separate_embedded_table_notes(source,candidate)
        selected=old._candidate_projection_line_indices(source,candidate)
        body=old._table_body_materialization_bbox(candidate,[annotation(a)for a in failed])
        expected=clean(dict(candidate=asdict(candidate),projection_line_indices=sorted(selected),body_bbox=body,calibrated_char_bboxes=repairs,native_chars=native))
        cases.append(dict(label=label,request=clean(request),expected=expected,split_note=len(candidate.annotations)>len(c['annotations']),expanded=bool(selected-set(candidate.line_indices)),repaired=bool(repairs)))
        return len(cases)-1
    def source(lines=(),chars=(),rules=()):return dict(page_size=[500.,700.],lines=list(lines),chars=list(chars),drawing_lines=list(rules))
    def line(i,b,text='body',angle=0,**kw):return dict(text=text,bbox=list(b),source_index=i,angle=angle,**kw)
    def candidate(b=(0.,0.,300.,200.),angle=0,indices=(),core=None,annotations=()):
        return dict(bbox=list(b),local_bbox=list(b),angle=angle,score=1.,core_bbox=None if core is None else list(core),line_indices=list(indices),annotations=list(annotations),inferred_grid=[],preserve_inline_font_styles=False,inferred_grid_authoritative=False)
    rng=random.Random(328171)
    for n in range(300):
        a=(0,90,180,270)[n%4];ls=[]
        for i in range(rng.randrange(0,24)):
            x=float(rng.randrange(5,290));y=float(rng.randrange(5,200));h=rng.choice([6.,10.,12.,16.]);b=(x,y,x+rng.choice([6.,10.,20.,35.]),y+h)
            rot=list(_rotate_bbox_from_upright(b,(500.,700.),a))
            ls.append(line(i*3+5,rot,angle=a if i%9 else 0,font_signature=rng.choice([None,['Arial',0],['Arial',64],['Serif',0]]),effective_height=rng.choice([0.,h]),em_height=rng.choice([0.,8.,12.]),style_scale_repaired=bool(n%3)))
        outer=_rotate_bbox_from_upright((0.,0.,300.,200.),(500.,700.),a);core=_rotate_bbox_from_upright((30.,40.,240.,160.),(500.,700.),a)if n%3 else None
        c=candidate(outer,a,[l['source_index']for l in ls[::7]],core)
        add(f'projection:{n}',source(ls),c)
    # Reverse source order forces repeated selected_lines snapshots to close a chain.
    for n in range(24):
        ls=[line(100-i,[10.+i*16.,40.,20.+i*16.,50.],font_signature=['Arial',0]if n%3 else None)for i in range(12)]
        if n%2:ls.reverse()
        if n%4==0:ls[3]['font_signature']=['Serif',0]
        c=candidate((0.,0.,250.,100.),indices=[100])
        add(f'iterative-baseline:{n}',source(ls),c)
    prefixes=['Note:','NOTES','Sources:','SourceX','Note_','note²','note١','note\u0301','ſources:','Ｎote:','注：','注释','说明:','for *','FOR\u00a0†','for*','* footnote','†\u2003x','‡','noteworthy']
    for n in range(160):
        prefix=prefixes[n%len(prefixes)];gap=[0.,1.,15.,15.00001][(n//20)%4]
        ls=[line(1,[10.,10.,30.,20.]),line(3,[10.,30.,30.,40.]),line(5,[10.,60.,80.,70.],text=prefix),line(7,[10.+(11. if n%7==0 else 0.),74.,80.,84.],text='continuation')]
        rules=[dict(bbox=[0.,y,100.,y],width=.2,orientation='horizontal')for y in(0.,25.,60.-gap,100.)]
        c=candidate((0.,0.,100.,100.),indices=[1,3,5,7])
        if n%13==0:c['annotations']=[dict(kind='footnote',bbox=[0.,85.,100.,95.],line_indices=[],line_bboxes={})]
        if n%17==0:c['angle']=90
        add(f'embedded-note:{n}',source(ls,rules=rules),c)
    # Annotation failure counts and union order are explicit helper-stage inputs.
    aa=[dict(kind=k,bbox=b,line_indices=[i],line_bboxes={str(i):b})for i,k,b in[(1,'caption',[0.,0.,300.,20.]),(2,'footnote',[0.,180.,300.,210.]),(3,'footnote',[-5.,160.,310.,185.])]]
    for core in(None,[10.,25.,290.,170.]):
        for fail in((),(aa[0],),(aa[1],),(aa[2],),(aa[0],aa[2]),tuple(aa)):
            add('annotation-body:'+str(core)+':'+str(len(fail)),source(),candidate(core=core,annotations=aa),fail)
    for n in range(100):
        count=[19,20,21,24][n%4];em=10.;loose=[21.99999,22.,30.][n%3];ink=[4.99999,5.,10.,14.,14.00001][n%5];cs=[]
        for i in range(count):
            b=[i*11.,20.,i*11.+em,20.+loose];tight=[i*11.,25.,i*11.+8.,25.+ink]
            cs.append(dict(char_idx=i,char='汉',bbox=b,tight_bbox=tight,origin=[b[0],40.],font=dict(name='ABCDEF+Serif',flags=0.,weight=400.)))
        cs.append(dict(char_idx=count,char='A',bbox=[0.,50.,8.,80.],tight_bbox=[0.,55.,8.,65.],origin=[0.,70.],font=dict(name='ABCDEF+Serif',flags=False,weight=400)))
        if n%7==0:cs[0]['font']['flags']=64
        if n%9==0:cs[1]['char']='㐀'  # Outside the exact U+4E00..9FFF anchor interval.
        calibrated={0:[0.,1.,10.,11.],count:[2.,3.,8.,9.]}if n%6==0 else None
        add(f'character-repair:{n}',source(chars=cs),candidate(),calibrated=calibrated)
    # Python min/max retain the selected operand's int/float type on bbox ties.
    for n in range(12 if typed else 0):
        s=source([line(1,[0,0,10,10]),line(2,[13,0,23,10])])
        c=candidate([0,0,100,100],indices=[1],core=[2,3,90,90])
        ann=dict(kind='footnote',bbox=[0,90.,100,110],line_indices=[3],line_bboxes={})
        c['annotations']=[ann]
        if n%3==0:c['annotations'].append(dict(kind='caption',bbox=[0,0,100,3],line_indices=[],line_bboxes={}))
        add(f'integer-types:{n}',s,c,[ann]if n%2 else())
    # Reuse independently frozen actual PDF inputs; lines are actual glyph visual rows.
    e=json.loads((ROOT/'build/migration/native-table-engine.json').read_text(encoding='utf-8'))
    for meta in e['real']:
        oldcase=e['cases'][meta['case_index']];ti=oldcase['request']['table_input'];r=oldcase['expected']['result'];ls=[]
        if r:
            groups={}
            for g in r['text']['glyphs']:groups.setdefault(g['visual_row'],[]).append(g)
            by={q['char_idx']:q for q in ti['chars']}
            for row,gs in groups.items():
                cs=[by[g['source_index']]for g in gs];bs=[c['bbox']for c in cs];b=[min(q[0]for q in bs),min(q[1]for q in bs),max(q[2]for q in bs),max(q[3]for q in bs)]
                ls.append(line(row,b,''.join(g['text']for g in gs),angle=ti['angle']))
        s=dict(page_size=ti['page_size'],lines=ls,chars=ti['chars'],drawing_lines=ti['drawing_lines'])
        c=candidate(ti['table_bbox'],ti['angle'],[l['source_index']for l in ls],ti['table_bbox'])
        i=add('real:'+meta['pdf']+':'+str(meta['angle']),s,c)
        real.append(dict(meta,case_index=i,rectangles=ti['rectangles']))
    prefix='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'
    files=['table_materialization.py','models.py','line_merging.py','line_layout.py','geometry.py','table_annotations.py','table_constants.py']
    if typed:cases=[c for c in cases if c['label'].startswith('integer-types:')];real=[]
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in files},cases=cases,real=real))
    print('Frozen',len(cases),'old block preparations;',sum(c['split_note']for c in cases),'note splits;',sum(c['expanded']for c in cases),'expanded claims;',sum(c['repaired']for c in cases),'repair cases;',len(real),'PDF/angle cases')
if __name__=='__main__':main()
