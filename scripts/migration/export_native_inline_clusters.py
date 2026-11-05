"""Freeze complete old 2D inline clusters and true multi-line cell role chains."""
import copy,json,random,sys
from dataclasses import fields,MISSING
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode
FIELDS='text bbox angle source_index source_bbox ink_bbox baseline chars visual_row_id run_index effective_height em_height font_signature font_coverage dominant_font_weight median_glyph_width leading_emphasis_width leading_typography_width paragraph_terminal caption_start native_math_words paragraph_formula_context split_from_row preserve_split_boundary semantic_type restored_inline_cluster compact_formula_cluster formula_candidate_only style_scale_repaired inline_math_regions paragraph_group title_band_id structural_title explicit_section_title title_suppressed native_title_label_left'.split()
def clean(v):
    from dataclasses import is_dataclass
    if is_dataclass(v):return clean({f.name:getattr(v,f.name)for f in fields(v)if not f.name.startswith('_')})
    if isinstance(v,(set,frozenset)):return sorted(v)
    if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    return encode(v)
def main():
    target=ROOT/'build/migration/native-inline-clusters.json';schema=ROOT/'rust/rapiddoc/src/processing/table/table_text_styles/cluster_defaults.json'
    if target.exists()or schema.exists():raise FileExistsError('immutable cluster reference exists')
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_merging as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _cell_script_roles,_render_styled_cell
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph,NativeTableCell
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    defaults={}
    for f in fields(_LineItem):
        if f.name not in FIELDS:continue
        if f.default is not MISSING:defaults[f.name]=clean(f.default)
        elif f.default_factory is not MISSING:defaults[f.name]=clean(f.default_factory())
    cases=[]
    def objects(raw):
        out=[]
        for v in copy.deepcopy(raw):
            if v.get('font_signature')is not None:v['font_signature']=tuple(v['font_signature'])
            if 'native_math_words'in v:v['native_math_words']=frozenset(v['native_math_words'])
            out.append(_LineItem(**v))
        return out
    def projection(l):return clean({k:getattr(l,k)for k in FIELDS})
    def add(label,raw,page=(400.,600.),tables=(),role=True,fraction=(),render=None,origins_override=None,tight_override=None):
        lines=objects(raw);identities={id(l):i for i,l in enumerate(lines)};groups={};events={}
        def profile(frame,event,arg):
            name=frame.f_code.co_name
            if event=='return'and name=='_merge_overlapping_inline_cluster':groups[id(arg)]=[identities[id(l)]for l,b in frame.f_locals['members']]
            elif event=='return'and name=='_has_paired_script_body_host'and arg:events['paired']=True
            elif event=='return'and name=='_overlapping_candidate_pairs'and arg is not None:events['indexed']=True
        sys.setprofile(profile)
        try:result=old._merge_overlapping_inline_text_clusters(lines,tuple(page),[tuple(t)for t in tables])
        finally:sys.setprofile(None)
        expected=dict(lines=[projection(l)for l in result],groups=[groups[id(l)]if id(l)in groups else[identities[id(l)]]for l in result])
        c=dict(label=label,request=clean(dict(lines=raw,page_size=list(page),table_bboxes=list(tables))),expected=expected,events=events)
        if role:
            chars=[q for l in lines for q in l.chars];tight={q['char_idx']:q['tight_bbox']for q in chars if type(q.get('char_idx'))is int and q.get('tight_bbox')is not None};origins={q['char_idx']:q['origin']for q in chars if type(q.get('char_idx'))is int and q.get('origin')is not None}
            if origins_override is not None:origins=origins_override
            if tight_override is not None:tight=tight_override
            roles=_cell_script_roles([],{},tuple(page),0,tight,origins,set(fraction),visual_lines=objects(raw))
            c.update(role_request=clean(dict(lines=raw,page_size=list(page),tight_bboxes=tight,origins=origins,fraction_members=list(fraction))),expected_roles=clean(roles))
            if render:
                cell,glyphs,height,by=render
                c['render_request']=clean(dict(cell=cell,glyphs=glyphs,median_height=height,chars_by_source=by,tight_bboxes=tight,origins=origins,preserve_font_styles=True))
                c['expected_html']=_render_styled_cell(NativeTableCell(**cell),[NativeTableGlyph(**g)for g in glyphs],height,roles,chars_by_source=by,tight_bboxes=tight,origins=origins,preserve_font_styles=True)
        cases.append(c)
    def line(i,text,b,a=0,h=10.,cov=1.,chars=None,**kw):
        return dict(text=text,bbox=list(_rotate_bbox_from_upright(tuple(b),(400.,600.),a)),angle=a,source_index=i,effective_height=h,font_coverage=cov,font_signature=['Arial',0],visual_row_id=i,chars=chars or [],**kw)
    def chars(text,x,y,h,start=0):
        return [dict(char=t,char_idx=start+i,bbox=[x+i*5.,y-h,x+i*5.+4.,y],tight_bbox=[x+i*5.,y-h,x+i*5.+4.,y],origin=[x+i*5.,y],font=dict(name='Arial',flags=0,weight=400.,size=h))for i,t in enumerate(text)]
    for n in range(80):
        a=(0,90,180,270)[n%4];ls=[line(10,'normal body',(10.,50.,80.,60.),a),line(20,'2',(80.,48.,84.,54.),a,h=6.,cov=.5)]
        if a==0:
            ls[0]['chars']=chars('normal body',10.,60.,10.)
            ls[1]['chars']=chars('2',65.,54.,6.,len(ls[0]['chars']))
        if n%5==1:ls[1]['visual_row_id']=10
        if n%5==2:ls[0]['caption_start']=True;ls[0]['bbox']=list(_rotate_bbox_from_upright((75.,50.,125.,60.),(400.,600.),a));ls[1]['bbox']=list(_rotate_bbox_from_upright((10.,48.,80.,54.),(400.,600.),a))
        if n%5==3:ls[1]['semantic_type']='paragraph_title'
        if n%5==4:ls[0]['style_scale_repaired']=True;ls[0]['em_height']=10.
        tables=[[50.,45.,60.,65.]]if n%11==0 and a==0 else []
        if n%2:ls.reverse()
        add(f'body:{n}',ls,tables=tables)
    for n in range(32):
        a=(0,90,180,270)[n%4]
        ls=[line(10+i,str(i),(50.+i,50.+i,64.+i,56.+i),a,h=6.,cov=.5)for i in range(3)]+[line(99,'wide distant body',(0.,300.,240.,310.),a)]
        if n%8==0:ls[0]['formula_candidate_only']=True
        if n%8==1:ls[0]['preserve_split_boundary']=True
        add(f'compact:{n}',ls)
    for n in range(36):
        h=chars('normal body A',10.,60.,10.);end=10.+len(h)*5.;h+=chars('x',end,54.,6.,len(h));sub=chars('y',end,66.,6.,len(h))
        ls=[line(10,'normal body Ax',(10.,46.,90.,66.),h=10.,cov=.6,chars=h,baseline=60.),line(20,'y',(10.,50.,90.,70.),h=10.,cov=.6,chars=sub,baseline=65.)]
        if n%6==1:ls[1]['preserve_split_boundary']=True
        if n%6==2:ls[0]['semantic_type']=ls[1]['semantic_type']='paragraph_title'
        if n%6==3:ls[1]['chars'][0]['origin'][1]=60.
        if n%6==4:
            ls[0]['source_index']=2**63-1;ls[1]['source_index']=2**63-2;ls[0]['visual_row_id']=1;ls[1]['visual_row_id']=2
        if n%6==5:ls[0]['chars'].insert(-1,dict(char='',char_idx=999,bbox=[0.,0.,1.,1.]))
        if n>=18:ls.reverse()
        add(f'paired:{n}',ls)
    for n in range(24):
        prefix=['A.','١)','12.','123.'][n%4]
        ls=[line(20,prefix,(10.,48.,20.,54.),h=6.,cov=.5,native_math_words=['x','sum'],paragraph_group=7,title_band_id=8,structural_title=True,explicit_section_title=True,title_suppressed=True),line(10,'Title body',(20.,50.,100.,60.),native_title_label_left=20.,native_math_words=['y','x'],paragraph_group=7,title_band_id=8,structural_title=True,explicit_section_title=n%2==0,title_suppressed=True)]
        if n%3==1:ls[0]['title_band_id']=9
        if n%3==2:ls[0]['paragraph_group']=None
        add(f'inherit:{n}',ls)
    for n in range(16):
        ls=[line(1,'left',(160.,50.,196.,60.),h=8.,cov=.5),line(2,'body',(204.,50.,246.,60.),h=10.,cov=1.)]
        ls[0]['style_scale_repaired']=bool(n%2);ls[0]['em_height']=8.
        if n%4==0:ls[0]['source_bbox']=[0.,50.,10.,60.]
        add(f'gutter:{n}',ls)
    rng=random.Random(817235)
    for n in range(240):
        ls=[];a=(0,90,180,270)[n%4]
        for i in range(rng.randrange(2,42)):
            x=float(rng.randrange(7)*12);y=float(rng.randrange(5)*6+20);w=rng.choice([6.,12.,40.,80.]);h=rng.choice([6.,10.,14.]);l=line(i,f'text {i}',(x,y,x+w,y+h),a,h=h,cov=rng.choice([.5,.75,1.]))
            l.update(visual_row_id=rng.choice([None,i//2,i]),formula_candidate_only=bool(n%13==0 and i%2),semantic_type=rng.choice([None,None,None,'paragraph_title']))
            ls.append(l)
        if n%3==0:rng.shuffle(ls)
        add(f'random:{n}',ls,role=n%3==0)
    f=json.loads((ROOT/'build/migration/table-script-preparation.json').read_text(encoding='utf-8'))
    for n,c in enumerate(f['preparation']):
        page=c['request']['table_input']['page_size'];by={}
        for i,q in enumerate(c['request']['table_input']['chars']):
            idx=q.get('char_idx',i)
            if type(idx)is int:by.setdefault(idx,q)
        for j,cell in enumerate(c['expected']['cells'][:4]):
            if not cell['visual_lines']:continue
            originals=cell['visual_lines'];origins={q['char_idx']:q['origin']for q in c['request']['table_input']['chars']if type(q.get('char_idx'))is int and q.get('origin')is not None}
            rawcell=next(q for q in c['request']['result']['cells']if(q['row'],q['col'])==(cell['row'],cell['col']))
            add(f'cell:{n}:{j}',originals,page=page,fraction=cell['fraction_members'],render=(rawcell,cell['glyphs'],c['request']['result']['text']['median_glyph_height'],by),origins_override=origins,tight_override={int(k):v for k,v in c['request']['tight_bboxes'].items()})
    add('empty',[])
    add('single',[line(0,'single',(10.,10.,100.,20.))])
    refs=['line_merging.py','line_layout.py','native_text.py','geometry.py','models.py','table_text_styles.py','_script_geometry.py']
    sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
    schema.write_text(json.dumps(defaults,ensure_ascii=False,indent=2),encoding='utf-8');write_json(target,dict(sources=sources,fields=FIELDS,defaults=defaults,cases=cases))
    print('Frozen',len(cases),'complete 2D merges;',sum(any(len(g)>1 for g in c['expected']['groups'])for c in cases),'merge-positive cases;',sum(c['events'].get('paired',False)for c in cases),'paired-positive;',sum(c['events'].get('indexed',False)for c in cases),'indexed;',sum(bool(c.get('expected_roles'))for c in cases),'script-positive;',sum('expected_html'in c for c in cases),'cell HTML chains')
if __name__=='__main__':main()
