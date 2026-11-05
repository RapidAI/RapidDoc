"""Freeze original plan application, character resplitting and evidence realignment."""
import copy,json,random
from dataclasses import asdict
from common import ROOT,write_json,digest
from export_form_runtime import typed
from export_native_late_image_text import project
from export_native_table_page_policies import empty

def lines(raw):return typed(dict(empty(),lines=raw)).lines
def plan_raw(plan):
    out=dict(document_style_anomaly=plan.document_style_anomaly,run_diagnostics=plan.run_diagnostics,style_inflated_runs=sorted(plan.style_inflated_runs))
    for key in ['line_repairs','char_repairs','line_style_scales','line_ink_bboxes','line_baselines']:
        out[key]=[]
        for (p,i),value in sorted(getattr(plan,key).items()):
            item=dict(page_index=p,**{'char_idx'if key=='char_repairs'else'source_index':i})
            if key.endswith('repairs'):item.update(asdict(value))
            else:item[{'line_style_scales':'scale','line_ink_bboxes':'bbox','line_baselines':'baseline'}[key]]=value
            out[key].append(item)
    out['local_metric_repair_sources']=[dict(page_index=p,source_index=i)for p,i in sorted(plan.local_metric_repair_sources)]
    return json.loads(json.dumps(out,ensure_ascii=False))
def load_plan(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.char_geometry import DocumentGeometryPlan,LineGeometryRepair,CharLayoutGeometry
    p=DocumentGeometryPlan();p.document_style_anomaly=raw.get('document_style_anomaly',False);p.run_diagnostics=copy.deepcopy(raw.get('run_diagnostics',[]));p.style_inflated_runs={tuple(k)for k in raw.get('style_inflated_runs',[])};p.local_metric_repair_sources={(e['page_index'],e['source_index'])for e in raw.get('local_metric_repair_sources',[])}
    for key in ['line_repairs','char_repairs','line_style_scales','line_ink_bboxes','line_baselines']:
        for e in raw.get(key,[]):
            e=copy.deepcopy(e);page=e.pop('page_index');i=e.pop('char_idx'if key=='char_repairs'else'source_index')
            if key.endswith('repairs'):
                for k in ['source_bbox','layout_bbox','ink_bbox','tight_bbox','origin','run_key']:
                    if e.get(k)is not None:e[k]=tuple(e[k])
                value=(CharLayoutGeometry if key=='char_repairs'else LineGeometryRepair)(**e)
            else:value=e[{'line_style_scales':'scale','line_ink_bboxes':'bbox','line_baselines':'baseline'}[key]]
            getattr(p,key)[page,i]=value
    return p
def apply_ref(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.char_geometry import apply_line_geometry_repairs
    ls=lines(q['lines']);apply_line_geometry_repairs(ls,page_index=q['page_index'],plan=load_plan(q['plan']),allow_y_trim=q['allow_y_trim']);return [project(l)for l in ls]
def evidence(raw,kind):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.types import PDFTextStyleLine,PDFTextLinkLine,PDFTextStyleRange,PDFTextLinkRange
    ranges='style_ranges'if kind=='style'else'link_ranges';cls=PDFTextStyleLine if kind=='style'else PDFTextLinkLine;rc=PDFTextStyleRange if kind=='style'else PDFTextLinkRange
    result=[]
    for item in raw:
        r=copy.deepcopy(item);r['bbox']=tuple(r['bbox']);r[ranges]=tuple(rc(**dict(x,styles=tuple(x['styles'])))if kind=='style'else rc(**x)for x in r[ranges]);result.append(cls(**r))
    return result
def resplit_ref(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _resplit_native_visual_runs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.matching import _realign_repaired_text_evidence
    source=lines(q['lines']);out,maps=_resplit_native_visual_runs(source,tuple(q['page_size']),{e['char_idx']:tuple(e['bbox'])for e in q['visual_bboxes']},source_index_start=q.get('source_index_start'))
    styles,links=_realign_repaired_text_evidence(evidence(q.get('style_lines',[]),'style'),evidence(q.get('link_lines',[]),'link'),{e['source_index']:tuple(e['bbox'])for e in q.get('line_bboxes',[])},maps)
    result=[]
    for i,r in sorted(maps.items()):
        positions={id(c):j for j,c in enumerate(r.source.chars)};result.append(dict(source_index=i,source=project(r.source),members=[project(l)for l in r.members],member_char_positions=[[positions[id(c)]for c in l.chars]for l in r.members]))
    return json.loads(json.dumps(dict(lines=[project(l)for l in out],source_lines=[project(l)for l in source],resplits=result,style_lines=[asdict(l)for l in styles],link_lines=[asdict(l)for l in links]),ensure_ascii=False))
def base():
    text='Left column native text Right column native text';chars=[];visual=[]
    for i,c in enumerate(text):
        x=30.+i*4.;vx=30.+i*5. if i<24 else 300.+(i-24)*5.
        chars.append(dict(char=c,char_idx=i,bbox=[x,200.,x+4.,210.],font=dict(name='Body',size=10.,flags=0,weight=400)))
        visual.append(dict(char_idx=i,bbox=[vx,200.,vx+5.,210.]))
    line=dict(text=text,bbox=[30.,190.,470.,215.],source_index=8,angle=0,chars=chars,visual_row_id=7,run_index=4,effective_height=10.,em_height=10.,baseline=210.,geometry_state='repair_x',geometry_confidence=.85)
    compact=''.join(c for c in text if not c.isspace());n=len(compact)
    return dict(lines=[project(lines([line])[0])],page_size=[500.,600.],visual_bboxes=visual,source_index_start=15,line_bboxes=[dict(source_index=8,bbox=line['bbox'])],style_lines=[dict(bbox=line['bbox'],text=compact,source_index=8,style_ranges=[dict(start=0,end=n,styles=['bold']),dict(start=10,end=n-5,styles=['italic'])])],link_lines=[dict(bbox=line['bbox'],text=compact,source_index=8,link_ranges=[dict(start=0,end=10,target='https://example.org/original')])])
def main():
    target=ROOT/'build/migration/native-geometry.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.char_geometry import DocumentGeometryPlan,LineGeometryRepair,build_document_geometry_plan
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPageTextGeometry
    from loguru import logger
    logger.remove();applied=[];resplit=[];actual=[]
    for state in ['healthy','repair_x','trim_y','repair_xy','uncertain']:
        for allow in [True,False]:
            for local in [True,False]:
                for scale in [None,9.]:
                    ls=base()['lines'];p=DocumentGeometryPlan();p.line_repairs[2,8]=LineGeometryRepair(source_bbox=(30.,180.,470.,230.),layout_bbox=(35.,195.,465.,215.),ink_bbox=(37.,198.,460.,213.),baseline=211.,em_height=10.,state=state,confidence=.8,split_y_candidate=True);p.line_ink_bboxes[2,8]=(34.,197.,462.,214.);p.line_baselines[2,8]=212.
                    if local:p.local_metric_repair_sources.add((2,8))
                    if scale is not None:p.line_style_scales[2,8]=scale
                    q=dict(lines=ls,plan=plan_raw(p),page_index=2,allow_y_trim=allow);applied.append(dict(label=f'{state}/{allow}/{local}/{scale}',request=q,expected=apply_ref(q)))
    for font in ['ABCDEF+Body','Body','B_o-d y','Different','Straße','STRASSE']:
        p=DocumentGeometryPlan();p.style_inflated_runs={('body',10.,0,400,0,'latin'),('strasse',10.,0,400,0,'latin')};ls=base()['lines'];ls[0]['font_signature']=[font,0];q=dict(lines=ls,plan=plan_raw(p),page_index=0,allow_y_trim=False);applied.append(dict(label='font/'+font,request=q,expected=apply_ref(q)))
    for mode in ['split','no-visual','empty','semantic','too-narrow','top-margin','bottom-margin','three-runs','not-columns','one-run','no-visible','blank','no-row','negative-next','no-next','reversed-chars','duplicate-indices','bool-index','newlines','ligature','soft-hyphen','unsafe-text','crossing-link','no-links','bbox-only','style-only','source-defaults']:
        q=base();l=q['lines'][0]
        if mode=='no-visual':q['visual_bboxes']=[]
        if mode=='empty':q['lines']=[]
        if mode=='semantic':l['semantic_type']='caption'
        if mode=='too-narrow':l['bbox']=[30.,190.,200.,215.]
        if mode=='top-margin':l['bbox']=[30.,0.,470.,20.]
        if mode=='bottom-margin':l['bbox']=[30.,580.,470.,600.]
        if mode=='three-runs':
            for i,r in enumerate(q['visual_bboxes']):r['bbox']=[30.+(i//16)*160+(i%16)*4,200.,34.+(i//16)*160+(i%16)*4,210.]
        if mode=='not-columns':
            for r in q['visual_bboxes'][24:]:r['bbox'][0]-=180.;r['bbox'][2]-=180.
        if mode=='one-run':
            for i,r in enumerate(q['visual_bboxes']):r['bbox']=[30.+i*5,200.,35.+i*5,210.]
        if mode=='no-visible':
            for c in l['chars']:c['char']='\x00'
            l['text']='  clean\t text  '
        if mode=='blank':l['chars']=[];l['text']=' '
        if mode=='no-row':l['visual_row_id']=None
        if mode=='negative-next':q['source_index_start']=-10
        if mode=='no-next':q.pop('source_index_start')
        if mode=='reversed-chars':l['chars'].reverse()
        if mode=='duplicate-indices':l['chars'][1]['char_idx']=0
        if mode=='bool-index':l['chars'][1]['char_idx']=True
        if mode=='newlines':l['chars'].insert(4,dict(char='\n',bbox=[0.,0.,1.,1.],char_idx=200))
        if mode in ['ligature','soft-hyphen']:
            l['chars'][2]['char']='ﬃ'if mode=='ligature'else'\u00ad'
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.common import _normalize_match_text
            for key in ['style_lines','link_lines']:q[key][0]['text']=_normalize_match_text(''.join(c['char']for c in l['chars']))
        if mode=='unsafe-text':q['style_lines'][0]['text']='unrelated evidence';q['link_lines'][0]['text']='unrelated evidence'
        if mode=='crossing-link':q['link_lines'][0]['link_ranges'][0].update(start=10,end=33)
        if mode=='no-links':q['link_lines'][0]['link_ranges']=[]
        if mode=='bbox-only':q['visual_bboxes']=[];q['line_bboxes'][0]['bbox']=[40.,195.,460.,215.]
        if mode=='style-only':q['link_lines']=[]
        if mode=='source-defaults':
            for k in ['baseline','em_height','geometry_state','geometry_confidence']:l[k]=project(lines([dict(text='x',bbox=[0.,0.,1.,1.],source_index=1,angle=0)])[0])[k]
        resplit.append(dict(label=mode,request=q,expected=resplit_ref(q)))
    f=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text(encoding='utf-8'));sources=f['request']['sources'];ls=[lines(s['lines'])for s in sources];geometries=[]
    for s in sources:
        chars=s['chars'];maps=lambda key:{c['char_idx']:tuple(c[key])for c in chars if c.get(key)is not None};geometries.append(PDFPageTextGeometry(chars=chars,tight_bboxes=maps('tight_bbox'),origins=maps('origin'),loose_bboxes=maps('loose_bbox')))
    p=build_document_geometry_plan(ls,geometries,[tuple(s['page_size'])for s in sources]);raw=plan_raw(p)
    for i,page_lines in enumerate(ls):
        for allow in [False,True]:
            q=dict(lines=[project(l)for l in page_lines],plan=raw,page_index=i,allow_y_trim=allow);applied.append(dict(label=f'actual-form/{i}/{allow}',request=q,expected=apply_ref(q)))
    write_json(target,dict(sources={'char_geometry.py':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/char_geometry.py').read_bytes())},apply=applied,resplit=resplit,actual_document_plan=raw));print(len(applied),'plan applications;',len(resplit),'resplit/evidence scenes;',sum(bool(c['expected']['resplits'])for c in resplit),'splits')
if __name__=='__main__':main()
