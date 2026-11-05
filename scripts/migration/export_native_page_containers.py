"""Freeze complete old index/Form selectors and their real continuous table-stage use."""
import copy,json,random
from dataclasses import replace
from types import SimpleNamespace as NS
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects
from export_native_table_page_commit import typed as basic_typed,reference as table_reference
from export_native_table_page_policies import empty
from export_native_code_containers import glyphs,base as code_base

def typed(raw):
    s=basic_typed(raw)
    for k in ['form_member_sources','form_path_sources']:
        setattr(s,k,{tuple(e['bbox']):frozenset(e['source_indices'])for e in raw.get(k,[])})
    s.form_infos=tuple(NS(**dict(f,bbox=tuple(f['bbox'])))for f in raw.get('form_infos',[]))
    return s

def project_maps(raw,s):
    raw=copy.deepcopy(raw)
    for k in ['form_member_sources','form_path_sources']:
        current=getattr(s,k)
        if k in raw or current:raw[k]=[dict(bbox=list(b),source_indices=sorted(ids))for b,ids in current.items()]
    return clean(raw)

def index_reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.index_blocks import _extract_index_blocks
    ls=objects(q['lines']);blocks,remaining=_extract_index_blocks(ls,tuple(q['page_size']),[tuple(b)for b in q.get('container_bboxes',[])],require_heading=q.get('require_heading',False),require_explicit_heading=q.get('require_explicit_heading',False))
    raw=copy.deepcopy(q['lines']);remaining_ids={id(l)for l in remaining};claimed=[];rest=[]
    for r,l in zip(raw,ls):
        if r.get('semantic_type')!=l.semantic_type:r['semantic_type']=l.semantic_type
        if id(l)in remaining_ids:rest.append(r)
        else:claimed.append(l.source_index)
    return clean(dict(lines=raw,index_blocks=blocks,remaining_lines=rest,claimed_line_indices=sorted(claimed)))

def forms_reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _select_form_image_bboxes
    s=typed(q['source']);boxes=_select_form_image_bboxes(s)
    return clean(dict(source=project_maps(q['source'],s),form_bboxes=boxes))

def page_reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _select_form_image_bboxes,_form_supersedes_nested_bbox,_detect_strong_graphic_bboxes,_graphic_members_are_numeric_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_aligned_numeric_tables,_detect_table_candidates
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_materialization import _materialize_table_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.code_blocks import _build_rule_delimited_code_blocks,_build_code_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_smaller,_bbox_overlap_in_first,_bbox_area
    from export_native_bar_graphics import MUTABLE,snapshot
    raw=copy.deepcopy(q['source'])
    for k in ['lines','chars','drawing_lines','path_infos','image_bboxes','signature_bboxes','form_bboxes']:raw.setdefault(k,[])
    ix=index_reference(dict(lines=raw['lines'],page_size=raw['page_size'],container_bboxes=raw['image_bboxes']+raw['signature_bboxes'],require_heading=True,require_explicit_heading=True));raw['lines']=ix['lines'];s=typed(raw);before=[tuple(getattr(l,k)for k in MUTABLE)for l in s.lines]
    prior=set(q.get('prior_claimed_line_indices',[]))|set(ix['claimed_line_indices']);protected=prior|{l.source_index for l in s.lines if l.semantic_type is not None};analysis=replace(s,lines=[l for l in s.lines if l.source_index not in protected]);forms=_select_form_image_bboxes(s)
    strong=_detect_strong_graphic_bboxes(analysis);numeric=_detect_aligned_numeric_tables(analysis,list(s.image_bboxes));strong=[b for b in strong if not any(_bbox_overlap_in_smaller(b,c.bbox)>=.5 and _bbox_area(b)>=.25*_bbox_area(c.bbox)for c in numeric)]
    excluded=strong+[tuple(b)for b in q.get('excluded_bboxes',[])];rule,rc=_build_rule_delimited_code_blocks(analysis,excluded+forms+s.image_bboxes+s.signature_bboxes,prior)
    separators={tuple(b)for b in q.get('table_header_separator_bboxes',[])};ts=replace(analysis,drawing_lines=[r for r in s.drawing_lines if tuple(r.bbox)not in separators]);candidates=[c for c in _detect_table_candidates(ts,excluded+[b['bbox']for b in rule])if not any(_form_supersedes_nested_bbox(f,c.bbox)for f in forms)]
    maps=lambda k:{int(i):v for i,v in(q.get(k)or{}).items()};tables,annotations,claimed=_materialize_table_blocks(s,candidates,tight_bboxes=maps('tight_bboxes'),origins=maps('origins'),calibrated_char_bboxes=maps('calibrated_char_bboxes')or None)
    for b in tables:b['_native_numeric_grid']=_graphic_members_are_numeric_grid([l for l in s.lines if _bbox_overlap_in_first(l.bbox,b['bbox'])>=.95])
    active=[f for f in forms if not any(_bbox_overlap_in_smaller(f,b['bbox'])>=.5 for b in tables)];claimed|=prior|rc
    filled,fc=_build_code_blocks(analysis,excluded+active+s.image_bboxes+s.signature_bboxes+[b['bbox']for b in tables],claimed);claimed|=fc
    raw['lines']=snapshot(raw['lines'],s.lines,before)
    for r,l in zip(raw['lines'],s.lines):
        if r.get('explicit_section_title',False)!=l.explicit_section_title:r['explicit_section_title']=l.explicit_section_title
    raw=project_maps(raw,s);remaining=[r for r,l in zip(raw['lines'],s.lines)if l.source_index not in claimed];fixed=copy.deepcopy(q.get('prior_fixed_blocks',[]))+rule+annotations+tables+filled+ix['index_blocks'];grid=_trailing_table_continuation_grid(s,NS(fixed_blocks=fixed,remaining_lines=[l for l in s.lines if l.source_index not in claimed]))
    return clean(dict(source=raw,strong_graphic_bboxes=strong,table_blocks=tables,annotation_blocks=annotations,claimed_line_indices=sorted(claimed),remaining_lines=remaining,fixed_blocks=fixed,table_bboxes=[b['bbox']for b in tables],next_table_grid=None if grid is None else dict(page_index=grid.page_index,angle=grid.angle,positions=grid.positions,tolerance=grid.tolerance),rule_code_blocks=rule,code_blocks=filled,index_blocks=ix['index_blocks'],selected_form_bboxes=forms,active_form_bboxes=active))

def index_base(kind='paged'):
    lines=[dict(text='Contents',bbox=[200.,60.,300.,80.],source_index=100,effective_height=20.,font_signature=['Bold',1],chars=[])]
    for i in range(7):lines.append(dict(text=f'{i+1}. Topic heading'+(f' ....... {i+1}'if kind=='leaders'else f' {i+1}'if kind=='paged'else''),bbox=[40.,100.+20*i,460.,110.+20*i],source_index=i,effective_height=10.,font_signature=['Body',0],chars=[],visual_row_id=i))
    if kind=='unpaged':
        for l in lines[1:]:l['bbox'][2]=180.
    return dict(lines=lines,page_size=[500.,600.],container_bboxes=[],require_heading=False,require_explicit_heading=False)

def form_base():
    s=empty();s['page_size']=[500.,600.];s['form_bboxes']=[[40.,80.,360.,360.]]
    s['lines']=[dict(text='label',bbox=[70.,120.+i*50,160.,130.+i*50],source_index=i,effective_height=10.,chars=[dict(char='x',char_idx=i,bbox=[80.,125.+i*50,90.,135.+i*50],tight_bbox=[80.,125.+i*50,90.,135.+i*50])])for i in range(2)]
    s['path_infos']=[dict(bbox=b,source_index=i,form_depth=1,segment_count=6,fill_visible=False,stroke_visible=True,rectangle_bboxes=[])for i,b in enumerate([[60.,100.,340.,110.],[60.,110.,70.,340.]])]
    s['drawing_lines']=[dict(bbox=b,orientation='horizontal',width=1.)for b in [[70.,340.,340.,340.],[70.,100.,340.,100.]]];s['chars']=[c for l in s['lines']for c in l['chars']];return s

def main():
    targets=[ROOT/f'build/migration/{name}.json'for name in ['native-index-containers','native-form-selection','native-container-table-page']]
    if any(p.exists()for p in targets):raise FileExistsError(targets)
    from loguru import logger
    logger.remove();index=[];forms=[];pages=[]
    def add(dest,label,q,f):dest.append(dict(label=label,request=copy.deepcopy(q),expected=f(q)))
    for kind in ['paged','leaders','unpaged']:
        for rh,re in [(False,False),(True,False),(True,True),(False,True)]:
            q=index_base(kind);q.update(require_heading=rh,require_explicit_heading=re);add(index,f'{kind}/{rh}/{re}',q,index_reference)
        for text in ['Table of Contents','目录','目 录','CONTENTS','Contentſ','TABLE OF CONTENTſ','Outline','1','', 'Contents\n','目\x1c录']:
            q=index_base(kind);q['lines'][0]['text']=text;add(index,kind+'/heading/'+repr(text),q,index_reference)
        for mode in ['no-heading','no-font','bad-font','numeric','no-page','right-low','left-high','short-width','gap','one-row','empty','protected','watermark','tight','sidecar','out-of-order']:
            q=index_base(kind);ls=q['lines']
            if mode=='no-heading':ls.pop(0)
            if mode=='no-font':
                for l in ls:l['font_signature']=None
            if mode=='bad-font':ls[-1]['font_signature']=['Other',1]
            if mode=='numeric':
                ls.pop(0)
                for l in ls:l['text']='1 222 333 444'
            if mode=='no-page':
                for l in ls[1:]:l['text']='Topic heading'
            if mode=='right-low':
                for l in ls[1:]:l['bbox'][2]=220.
            if mode=='left-high':
                for l in ls[1:]:l['bbox'][0]=180.
            if mode=='short-width':
                for l in ls[1:]:l['bbox'][0]=250.;l['bbox'][2]=460.
            if mode=='gap':ls[-3]['bbox'][1]+=90.;ls[-3]['bbox'][3]+=90.
            if mode=='one-row':
                for l in ls[1:]:l['visual_row_id']=0
            if mode=='empty':q['lines']=[]
            if mode=='protected':
                for l in ls[1:4]:l['semantic_type']='caption'
            if mode=='watermark':
                for l in ls[1:]:l['bbox'][3]=l['bbox'][1]+50.
            if mode=='tight':
                for l in ls[1:]:l['ink_bbox']=[l['bbox'][0]+2.,l['bbox'][1]+2.,l['bbox'][2]-2.,l['bbox'][3]-2.]
            if mode=='sidecar':
                more=[]
                for l in ls[1:]:l['text']=l['text'].rsplit(' ',1)[0];l['bbox'][2]=200.;more.append(dict(l,text=str(l['source_index']+1),bbox=[450.,l['bbox'][1],460.,l['bbox'][3]],source_index=l['source_index']+1000))
                ls+=more
            if mode=='out-of-order':ls.reverse()
            add(index,kind+'/'+mode,q,index_reference)
        for suffix in ['i','IV','xiv','Ⅳ','１２','١٢','a1','a iv','1iv','中iv','1.','2）','３\u0085',']','1\x1c']:
            q=index_base(kind)
            for l in q['lines'][1:]:l['text']='Topic '+suffix
            add(index,kind+'/page/'+repr(suffix),q,index_reference)
    rng=random.Random(271028)
    for n in range(75):
        q=index_base(rng.choice(['paged','leaders','unpaged']));q['require_heading']=bool(n%2);q['require_explicit_heading']=bool(n%3==0)
        for l in q['lines'][1:]:
            if rng.random()<.25:l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-5.,5.])
        if n%4==0:q['container_bboxes']=[[30.,90.,470.,260.]]
        rng.shuffle(q['lines']);add(index,'random/'+str(n),q,index_reference)
    for mode in ['base','no-lines','no-paths','one-path','one-row','tiny','large','retained','reversed','clipped','zero','angle','style-scale','members-empty','members-valid','members-outsider','source-indices','tight-edge','missing-tight','missing-ink','paths-empty','paths-selected','alias']:
        s=form_base();b=s['form_bboxes'][0]
        if mode=='no-lines':s['lines']=[]
        if mode=='no-paths':s['path_infos']=[]
        if mode=='one-path':s['path_infos']=s['path_infos'][:1]
        if mode=='one-row':
            for l in s['lines']:l['visual_row_id']=0
            s['drawing_lines']=[]
        if mode=='tiny':s['form_bboxes']=[[40.,80.,50.,90.]]
        if mode in ['large','retained']:s['form_bboxes']=[[0.,0.,500.,600.]]
        if mode=='retained':s['form_infos']=[dict(bbox=[0.,0.,500.,600.])]
        if mode=='reversed':s['form_bboxes']=[[360.,360.,40.,80.]]
        if mode=='clipped':s['form_bboxes']=[[-20.,-10.,360.,360.]]
        if mode=='zero':s['page_size']=[0.,600.]
        if mode=='angle':
            for l in s['lines']:l['angle']=90
        if mode=='style-scale':
            for l in s['lines']:l.update(style_scale_repaired=True,em_height=100.)
        if mode.startswith('members-')or mode in ['source-indices','alias']:
            s['form_member_sources']=[dict(bbox=b,source_indices=[]if mode=='members-empty'else[0,1]if mode!='members-outsider'else[99])]
        if mode=='source-indices':
            for l in s['lines']:
                for ch in l['chars']:ch['source_indices']=[99,ch['char_idx']];ch['char_idx']=99
        if mode in ['tight-edge','missing-tight','missing-ink']:
            for l in s['lines']:l['bbox'][0]=-100.;l['bbox'][2]=600.
        if mode=='missing-tight':
            for l in s['lines']:
                for ch in l['chars']:ch.pop('tight_bbox')
        if mode=='missing-ink':
            for l in s['lines']:l['ink_bbox']=[80.,125.,90.,135.]
        if mode in ['paths-empty','paths-selected','alias']:s['form_path_sources']=[dict(bbox=b,source_indices=[]if mode=='paths-empty'else[0,1])]
        add(forms,mode,dict(source=s),forms_reference)
    for n in range(65):
        s=form_base();s['form_bboxes'][0][rng.randrange(4)]+=rng.choice([-.001,.001,-50.,50.]);s['form_bboxes']*=1+n%3
        if n%3==0:s['path_infos']=[]
        if n%4==0:s['drawing_lines']=[]
        add(forms,'random/'+str(n),dict(source=s),forms_reference)
    for c in index[::9]:
        s=empty();s['lines']=c['request']['lines'];s['page_size']=c['request']['page_size'];s['image_bboxes']=c['request']['container_bboxes'];add(pages,'index/'+c['label'],dict(source=s),page_reference)
    for c in forms[::3]:add(pages,'form/'+c['label'],c['request'],page_reference)
    for name in ['native-code-table-grid-mixtures','native-code-table-page']:
        for c in json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases'][::5]:add(pages,name+'/'+c['label'],c['request'],page_reference)
    for kind in ['filled','ruled','algorithm']:
        q=code_base(kind);q['source']['form_bboxes']=[[20.,80.,380.,260.]];add(pages,'form-protects-code/'+kind,q,page_reference)
    refs={p:digest((ROOT/p).read_bytes())for p in ['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/index_blocks.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/form_roles.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py']}
    for p,cases in zip(targets,[index,forms,pages]):
        write_json(p,dict(sources=refs,cases=cases,errors=[]));print(p.name,len(cases),'positive',sum(bool(c['expected'].get('index_blocks')or c['expected'].get('form_bboxes')or c['expected'].get('table_blocks')or c['expected'].get('code_blocks')or c['expected'].get('rule_code_blocks'))for c in cases))

if __name__=='__main__':main()
