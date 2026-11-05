"""Freeze actual old complete block transactions and media, without overriding stages."""
import copy,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    cache_only='--cache-only'in sys.argv
    target=ROOT/('build/migration/native-table-block-commit-cache.json'if cache_only else'build/migration/native-table-block-commit.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_PageSource,_AxisLine,_TableCandidate,_TableAnnotation,PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_materialization as old
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    from loguru import logger
    logger.remove()
    cases=[];real=[];excluded_engine=[]
    def source(v):
        lines=[]
        for raw in copy.deepcopy(v['lines']):
            raw['bbox']=tuple(raw['bbox'])
            if raw.get('font_signature')is not None:raw['font_signature']=tuple(raw['font_signature'])
            lines.append(_LineItem(**raw))
        paths=[]
        for i,r in enumerate(v.get('path_infos',[])):
            r=copy.deepcopy(r);r.update(bbox=tuple(r['bbox']));r.setdefault('source_index',i);r.setdefault('form_depth',0);paths.append(PDFPathInfo(**r))
        return _PageSource(tuple(v['page_size']),lines,copy.deepcopy(v['chars']),[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in v['drawing_lines']],image_bboxes=[tuple(b)for b in v.get('image_bboxes',[])],form_bboxes=[tuple(b)for b in v.get('form_bboxes',[])],path_infos=paths,page_index=v.get('page_index'))
    def candidate(v):
        v=copy.deepcopy(v)
        for k in ['bbox','core_bbox','local_bbox']:
            if v[k]is not None:v[k]=tuple(v[k])
        v['line_indices']=set(v['line_indices']);v['annotations']=[_TableAnnotation(a['kind'],tuple(a['bbox']),set(a['line_indices']),{int(k):tuple(b)for k,b in a['line_bboxes'].items()})for a in v['annotations']]
        v['inferred_grid']=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in v['inferred_grid']]
        return _TableCandidate(**v)
    def add(label,q):
        s=source(q['source']);cs=[candidate(c)for c in q['candidates']];events={}
        def profile(frame,event,arg):
            if event!='return':return
            n=frame.f_code.co_name
            if n=='_recover_native_table_html':events['recovery_error']=events.get('recovery_error',False)or arg is None;events['native']=events.get('native',False)or bool(arg)
            if n=='project_pdf_table_text':events['projection']=True
            if n=='_restore_front_matter_text_panel'and arg:events['front_restore']=True
            if n=='_materialize_table_annotations'and arg:events['provisional_annotations']=events.get('provisional_annotations',0)+len(arg[0])
            if n=='attach_table_media'and arg:events['media']=True
        tight=q.get('tight_bboxes')or{};origins=q.get('origins')or{}
        tight={int(k):b for k,b in tight.items()};origins={int(k):b for k,b in origins.items()}
        calibrated={int(k):b for k,b in q.get('calibrated_char_bboxes',{}).items()}if q.get('calibrated_char_bboxes')else None
        sys.setprofile(profile)
        try:tables,annotations,claimed=old._materialize_table_blocks(s,cs,tight_bboxes=tight,origins=origins,calibrated_char_bboxes=calibrated)
        finally:sys.setprofile(None)
        for b in annotations:b['_font_signatures']=sorted(b['_font_signatures'])
        lines=copy.deepcopy(q['source']['lines'])
        for raw,l in zip(lines,s.lines):
            if raw.get('semantic_type')!=l.semantic_type or raw.get('explicit_section_title',False)!=l.explicit_section_title:raw.update(semantic_type=l.semantic_type,explicit_section_title=l.explicit_section_title)
        events['annotation_rollback']=bool(events.get('provisional_annotations')and not tables and not annotations)
        cases.append(dict(label=label,request=clean(q),expected=clean(dict(table_blocks=tables,annotation_blocks=annotations,claimed_line_indices=sorted(claimed),lines=lines,candidates=[asdict(c)for c in cs])),events=events))
        return len(cases)-1
    def wrap(q):
        r=copy.deepcopy(q);r.pop('failed_annotations',None);r['candidates']=[r.pop('candidate')];return r
    a=json.loads((ROOT/'build/migration/native-table-annotations.json').read_text(encoding='utf-8'))
    for c in a['cases'][::3]:add('annotations:'+c['label'],wrap(c['request']))
    for c in a['front']:
        q=wrap(a['cases'][0]['request']);q['source']=c['request']['source'];q['candidates'][0].update(bbox=c['request']['bbox'],core_bbox=c['request']['bbox'],local_bbox=c['request']['bbox'],angle=0,annotations=[],line_indices=[l['source_index']for l in q['source']['lines']]);add(c['label'],q)
    e=json.loads((ROOT/'build/migration/native-table-engine.json').read_text(encoding='utf-8'))
    for n,c in enumerate(e['cases'][::4]):
        ti=c['request']['table_input']
        if any(not isinstance(char.get('char'),str)for char in ti['chars']):excluded_engine.append(dict(case_index=n*4,reason='not a typed PageSource Char string'));continue
        q=wrap(a['cases'][0]['request']);q['source'].update(chars=ti['chars'],drawing_lines=ti['drawing_lines'],page_size=ti['page_size'],path_infos=ti['rectangles'],lines=[],page_index=1)
        q['candidates'][0].update(bbox=ti['table_bbox'],local_bbox=ti['table_bbox'],core_bbox=None,angle=ti['angle'],line_indices=[],annotations=[])
        add(f'engine:{n*4}',q)
    for meta in a['real']:
        q=wrap(a['cases'][meta['case_index']]['request']);q['source']['path_infos']=meta['rectangles'];q['source']['page_index']=meta['page_index']
        q['tight_bboxes']={c['char_idx']:c.get('tight_bbox')for c in q['source']['chars']};q['origins']={c['char_idx']:c.get('origin')for c in q['source']['chars']}
        i=add('real:'+str(meta['case_index']),q);real.append(dict(meta,case_index=i))
    # Multiple candidates: successful full-box de-duplication, stable score ties, spatial output order.
    native=next(copy.deepcopy(c['request'])for c in cases if c['events'].get('native'))
    for n in range(16):
        q=copy.deepcopy(native);c=q['candidates'][0];other=copy.deepcopy(c);other['score']=c['score']+(1. if n%2 else 0.);other['annotations']=[]
        if n%3==0:other['bbox']=[0.,0.,2.,2.];other['core_bbox']=None
        q['candidates']=[c,other,copy.deepcopy(c)]
        add(f'multiple:{n}',q)
    # Actual vector grids with cells containing media; counts/topology stay independently recovered.
    for n in range(24):
        q=copy.deepcopy(native);t=q['candidates'][0]['bbox'];w=(t[2]-t[0])/12;h=(t[3]-t[1])/12
        b=[t[0]+w,t[1]+h,t[0]+2*w,t[1]+2*h]
        q['source']['image_bboxes']=[b,b.copy()]if n%3 else[b,[t[0],t[1],t[2],t[3]]]
        if n%4==0:q['source']['form_bboxes']=[b]
        if n%5==0:q['source'].setdefault('path_infos',[]).append(dict(bbox=b,segment_count=6,fill_visible=True,stroke_visible=True,form_depth=1,source_index=999))
        add(f'media:{n}',q)
    # Unusable side maps exercise real recovery/render exceptions, not substituted stage outputs.
    for n in range(4):
        q=copy.deepcopy(native);q['tight_bboxes']={str(c.get('char_idx',i)):c['bbox']for i,c in enumerate(q['source']['chars'])};q['origins']={k:[0.]if n%2 else'bad'for k in q['tight_bboxes']}
        add(f'recovery-error:{n}',q)
    if cache_only:
        q=wrap(a['cases'][0]['request']);q['source'].update(lines=[],drawing_lines=[],page_index=1,chars=[dict(char='A',char_idx=i,bbox=[400.,400.,401.,401.],font=dict(name=f'Font{i}'))for i in range(2000)])
        q['candidates']=[]
        for i in range(8):
            c=copy.deepcopy(a['cases'][0]['request']['candidate']);b=[20.*i,0.,20.*i+10.,10.];c.update(bbox=b,local_bbox=b,core_bbox=None,score=float(i),annotations=[],line_indices=[]);q['candidates'].append(c)
            q['source']['chars'].append(dict(char=f'B{i}',char_idx=2000+i,bbox=[20.*i+1.,1.,20.*i+2.,2.]))
        add('cache:global-calibration-once',q)
        cases=[c for c in cases if c['label'].startswith('cache:')];real=[]
    paths=['table_materialization.py','table_media.py','spatial_text.py','_table_recovery/engine.py','table_text_styles.py']
    prefix='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in paths},cases=cases,real=real,excluded_engine=excluded_engine))
    print('Frozen',len(cases),'old complete block transactions;',sum(c['events'].get('native',False)for c in cases),'native;',sum(c['events'].get('projection',False)for c in cases),'projection;',sum(c['events'].get('front_restore',False)for c in cases),'front restored;',sum(c['events'].get('annotation_rollback',False)for c in cases),'annotation rollbacks;',sum(c['events'].get('media',False)for c in cases),'media;',sum(c['events'].get('recovery_error',False)for c in cases),'recovery errors')
if __name__=='__main__':main()
