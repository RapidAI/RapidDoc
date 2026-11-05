"""Freeze full closed-rule branch, source preparation and connected-grid expansion together."""
import copy,random,json,sys
from dataclasses import asdict
from types import SimpleNamespace as NS
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects as line_objects
from export_native_table_candidate_merge import objects as candidate_objects,encode,candidate
def dumped(c):
    raw=asdict(c);raw["line_indices"]=set(c.line_indices);return encode(raw)

def setup(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _transform_axis_lines,_rotate_bbox_to_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    source=q['source'];angle=q.get('angle',0);size=tuple(source['page_size']);lines=[l for l in line_objects(source.get('lines',[]))if l.angle==angle];fs=old._build_fragments(lines,size);em=q.get('median_height');em=old._median_fragment_height(fs)if em is None else em;rows=old._cluster_fragment_rows(fs,em);axis=_transform_axis_lines([_AxisLine(tuple(r['bbox']),r.get('width',0.),r.get('orientation','horizontal'))for r in source.get('drawing_lines',[])],size,angle);excluded=[_rotate_bbox_to_upright(tuple(b),size,angle)for b in q.get('excluded_bboxes',[])];return old,source,size,angle,lines,rows,axis,em,excluded
def references(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_union_many,_rotate_bbox_to_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_annotations import _table_caption_candidates
    old,source,size,angle,lines,rows,axis,em,excluded=setup(q);components=old._connected_rule_grid_components(axis,em)
    inspected=clean(dict(median_height=em,rows=[asdict(r)for r in rows],components=[dict(axis_line_indices=[next(i for i,a in enumerate(axis)if a is r)for r in group],bbox=_bbox_union_many([r.bbox for r in group]),vertical_positions=old._closed_grid_vertical_track_positions(group,axis,em))for group in components],caption_candidates=[dict(source_index=l.source_index,local_bbox=b)for l,b in _table_caption_candidates(lines,size,angle,em)]))
    previous=NS(**source['previous_table_grid'])if source.get('previous_table_grid')is not None else None
    obstacles=[_rotate_bbox_to_upright(tuple(b),size,angle)for b in q.get('excluded_bboxes',[])+source.get('image_bboxes',[])+source.get('signature_bboxes',[])+source.get('form_bboxes',[])];forms=[_rotate_bbox_to_upright(tuple(b),size,angle)for b in source.get('form_bboxes',[])];cs=old._build_closed_rule_grid_candidates(rows,lines,size,angle,em,axis,excluded,previous_table_grid=previous,page_index=source.get('page_index'),leading_obstacles=obstacles,form_bboxes=forms)
    detected=clean(dict(median_height=em,candidates=[dumped(c)for c in cs]));expanded=old._expand_candidates_to_connected_rule_grids(candidate_objects(q.get('candidates',[])),rows,size,angle,em,axis,excluded)
    return inspected,detected,clean(dict(candidates=[dumped(c)for c in expanded]))
def base():
    rules=[dict(bbox=[40.,y,240.,y],orientation='horizontal',width=0.)for y in [100.,140.,180.,220.]]+[dict(bbox=[x,100.,x,220.],orientation='vertical',width=0.)for x in [40.,140.,240.]]
    lines=[dict(text='cell',bbox=[x,y,x+30.,y+10.],source_index=i,angle=0,chars=[],effective_height=10.,visual_row_id=None)for i,(y,x)in enumerate((y,x)for y in [110.,150.,190.]for x in [60.,160.])]
    lines+=[dict(text='Table 1: Results',bbox=[40.,75.,200.,85.],source_index=8,angle=0,chars=[],effective_height=10.),dict(text='Note: measured values',bbox=[40.,225.,200.,235.],source_index=9,angle=0,chars=[],effective_height=10.)]
    return dict(source=dict(page_size=[400.,600.],lines=lines,drawing_lines=rules,form_bboxes=[],image_bboxes=[],signature_bboxes=[]),angle=0,median_height=10.,excluded_bboxes=[],candidates=[candidate([40.,140.,240.,180.],score=80.,ids=[2,3],core=[40.,140.,240.,180.])])
def main():
    targets=[ROOT/f'build/migration/{name}.json'for name in ['native-rule-page','native-closed-rule-tables','native-connected-table-grids']]
    if '--resume-expansion'in sys.argv:
        if targets[2].exists():raise FileExistsError(targets[2])
        frozen=json.loads(targets[0].read_text(encoding='utf-8'));cases=[dict(label=c['label'],request=c['request'],expected=references(c['request'])[2])for c in frozen['cases']]
        write_json(targets[2],dict(sources=frozen['sources'],cases=cases));print('Frozen remaining expansion reference',len(cases));return
    if any(p.exists()for p in targets):raise FileExistsError('immutable closed-rule reference exists')
    corpora=[[],[],[]]
    def add(label,q=None):
        q=clean(copy.deepcopy(q or base()));expected=references(q)
        for dest,result in zip(corpora,expected):dest.append(dict(label=label,request=q,expected=result))
    add('base')
    q=base();q['median_height']=None;add('automatic-scale',q)
    for count in [0,1,2,3,4]:q=base();q['source']['drawing_lines']=q['source']['drawing_lines'][:count]+q['source']['drawing_lines'][4:];add('horizontal-count-'+str(count),q)
    for count in [0,1,2,3]:q=base();q['source']['drawing_lines']=q['source']['drawing_lines'][:4]+q['source']['drawing_lines'][4:4+count];add('vertical-count-'+str(count),q)
    for marker in ['Table 1: Results','Table1: Results','Table 1 shows values','TABLE Ⅰ: Results','表一 结果','表一：结果','Table ١: Results','Tab. 1. Results','Table1中文','TABLE IX: Results','Table','Tab...']:
        q=base();q['source']['lines'][-2]['text']=marker;add('caption-'+repr(marker),q)
    q=base();q['source']['lines'][-2]['text']='Table';q['source']['lines'].append(dict(text='1.',bbox=[205.,75.,215.,85.],source_index=10,angle=0,chars=[],effective_height=10.));add('split-caption-peer',q)
    for delta in [.99999,1.,1.00001]:q=base();q['source']['drawing_lines'].insert(1,dict(bbox=[40.,100.+delta,240.,100.+delta],orientation='horizontal',width=0.));add('duplicate-horizontal-'+str(delta),q)
    for delta in [2.49999,2.5,2.50001]:q=base();q['source']['drawing_lines'][4]['bbox'][3]=140.-delta;add('track-gap-'+str(delta),q)
    q=base();q['source']['drawing_lines'][4:]=[dict(bbox=[x,100.,x,220.],orientation='vertical',width=0.)for x in [100.,180.]];add('interior-tracks-not-outer',q)
    q=base();q['excluded_bboxes']=[[35.,95.,245.,225.]];add('exclude-grid',q)
    q=base();q['excluded_bboxes']=[[50.,105.,100.,125.]];add('exclude-fragments-after-centre',q)
    q=base();q['source']['form_bboxes']=[[0.,0.,400.,600.]];add('containing-form',q)
    q=base();q['source']['form_bboxes']=[[60.,110.,80.,120.]];add('small-form-not-parent',q)
    for mode in ['no-previous','previous','wrong-page','wrong-columns','obstacle','prose-above']:
        q=base();q['source']['drawing_lines']=[q['source']['drawing_lines'][0],q['source']['drawing_lines'][3],*q['source']['drawing_lines'][4:]];q['source']['lines']=[l for l in q['source']['lines'][:6]if l['bbox'][0]<100.];q['source']['page_index']=1
        if mode!='no-previous':q['source']['previous_table_grid']=dict(page_index=0,angle=0,positions=[.1,.35,.6],tolerance=.005)
        if mode=='wrong-page':q['source']['page_index']=2
        if mode=='wrong-columns':q['source']['previous_table_grid']['positions']=[.1,.3,.6]
        if mode=='obstacle':q['source']['image_bboxes']=[[40.,50.,80.,70.]]
        if mode=='prose-above':q['source']['lines'].append(dict(text='above',bbox=[40.,50.,100.,60.],source_index=20,angle=0,chars=[],effective_height=10.))
        add('single-row-continuation-'+mode,q)
    q=base();q['candidates'][0]['annotations']=[dict(kind='caption',bbox=[40.,110.,90.,120.],line_indices=[0],line_bboxes=[[0,[40.,110.,90.,120.]]])];add('expanded-members-exclude-annotation',q)
    q=base();q['candidates'][0]['core_bbox']=None;add('no-core-not-expanded',q)
    q=base();q['candidates']*=2;add('duplicate-candidates',q)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    for angle in [0,90,180,270,15]:
        q=base();q['angle']=angle
        for l in q['source']['lines']:l['angle']=angle;l['bbox']=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['source']['page_size']),angle)
        for r in q['source']['drawing_lines']:r['bbox']=_rotate_bbox_from_upright(tuple(r['bbox']),tuple(q['source']['page_size']),angle)
        for c in q['candidates']:
            for key in ['bbox','core_bbox']:c[key]=_rotate_bbox_from_upright(tuple(c[key]),tuple(q['source']['page_size']),angle)
            c['angle']=angle
        add('angle-'+str(angle),q)
    rng=random.Random(271001)
    for i in range(80):
        q=base();rng.shuffle(q['source']['lines']);rng.shuffle(q['source']['drawing_lines']);r=rng.choice(q['source']['drawing_lines']);r['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-2.,2.,2.501]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_constants.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py'];sources={p:digest((ROOT/p).read_bytes())for p in refs}
    for p,cases in zip(targets,corpora):write_json(p,dict(sources=sources,cases=cases));print('Frozen',p.name,len(cases))
    print('Closed candidate positives',sum(bool(c['expected']['candidates'])for c in corpora[1]),'; expanded changes',sum(c['request']['candidates']!=c['expected']['candidates']for c in corpora[2]))
if __name__=='__main__':main()
