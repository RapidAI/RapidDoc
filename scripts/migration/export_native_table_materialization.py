"""Freeze complete old table HTML and recovery; never override existing evidence."""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/native-table-materialization.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableResult,NativeTableText,NativeTableGlyph,NativeTableCell
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import render_native_table_html_with_scripts as render
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_materialization import recover_table_result as recover
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    renders=[];recoveries=[];real=[]
    def inp(v):
        return NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v['chars']),tuple(NativeTableRule(**(q|dict(bbox=tuple(q['bbox']))))for q in v.get('drawing_lines',[])),tuple(NativeTableRectangle(**(q|dict(bbox=tuple(q['bbox']))))for q in v.get('rectangles',[])))
    def res(r):
        t=r['text']
        if isinstance(t,dict)and 'glyphs'in t:t=NativeTableText(tuple(NativeTableGlyph(**g)for g in t['glyphs']),tuple(t['rows']),t['median_glyph_width'],t['median_glyph_height'])
        return NativeTableResult(r['html'],r['rows'],r['cols'],tuple(NativeTableCell(**c)for c in r['cells']),t,r['source'],r['confidence'],tuple(r.get('diagnostics',[])))
    def side(v):
        cs=v['chars'];tight={};origins={}
        for i,c in enumerate(cs):
            k=c.get('char_idx',i)
            if type(k)is not int:continue
            tight.setdefault(k,c.get('tight_bbox',c.get('bbox')));origins.setdefault(k,c.get('origin',None))
        return tight,origins
    def add(label,v,r,tight,origins,preserve):
        request=dict(table_input=v,result=r,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve)
        expected=render(res(r),inp(v),tight,origins,preserve_font_styles=preserve)
        renders.append(dict(label=label,request=copy.deepcopy(request),expected=expected))
    e=json.loads((ROOT/'build/migration/native-table-engine.json').read_text(encoding='utf-8'))
    for i,c in enumerate(e['cases']):
        v=c['request']['table_input'];tight,origins=side(v)
        r=recover(inp(v),tight,origins,preserve_font_styles=False)
        expected=None if r is None else dict(content=r[0],result=asdict(r[1]))
        recoveries.append(dict(label=f'engine:{i}',request=dict(table_input=v,tight_bboxes=tight,origins=origins,preserve_font_styles=False),expected=expected))
    for c in json.loads((ROOT/'build/migration/table-script-preparation.json').read_text(encoding='utf-8'))['preparation']:
        q=c['request'];v=q['table_input'];r=q['result'];tight=q['tight_bboxes'];_,origins=side(v)
        # Frozen JSON string keys become the old API's integer source keys.
        tight={int(k):b for k,b in tight.items()}
        for preserve in (False,True):add('prepared:'+c['label']+':'+str(preserve),v,r,tight,origins,preserve)
    preparations=json.loads((ROOT/'build/migration/table-script-preparation.json').read_text(encoding='utf-8'))['preparation']
    for i,c in enumerate(json.loads((ROOT/'build/migration/table-script-preparation-extra.json').read_text(encoding='utf-8'))['chains']):
        if not c['expected_roles']:continue
        q=preparations[c['preparation_index']]['request']
        add(f'automatic-script:{i}',q['table_input'],q['result'],{int(k):b for k,b in q['tight_bboxes'].items()},{int(k):b for k,b in c['origins'].items()},c['preserve_font_styles'])
    # Genuine missing-space, preserved font, escaped content and exact early fallback controls.
    v=dict(table_bbox=[0.,0.,200.,100.],page_size=[200.,100.],angle=0,drawing_lines=[],rectangles=[],chars=[])
    gs=[]
    for i,t in enumerate('AB'):
        b=[10.+i*9,20.,18.+i*9,30.];ink=[11.+i*9,20.,14.+i*9,30.]
        v['chars'].append(dict(char_idx=i,char=t,bbox=b,tight_bbox=ink,origin=[b[0],30.],writing_angle=0.,font=dict(name='Arial-BoldItalic',flags=0,weight=700.,size=10.)))
        gs.append(asdict(NativeTableGlyph(i,i,t,tuple(b),0)))
    r=dict(html='SENTINEL<&>',rows=1,cols=1,cells=[asdict(NativeTableCell(0,0,2,3,(0.,0.,200.,100.),'AB',(0,1)))],text=dict(glyphs=gs,rows=[],median_glyph_width=8.,median_glyph_height=10.),source='vector_grid',confidence=1.,diagnostics=[])
    tight,origins=side(v)
    for preserve in (False,True):
        add('tight-space:'+str(preserve),v,r,tight,origins,preserve)
        for a,b in [(None,origins),(tight,None),({},origins),(tight,{})]:add('missing-map:'+str(preserve),v,r,a,b,preserve)
        for t in [None,{}]:q=copy.deepcopy(r);q['text']=t;add('missing-text:'+str(preserve),v,q,tight,origins,preserve)
        q=copy.deepcopy(r);q['cells'][0]['content']='wrong<&>"';add('mismatch:'+str(preserve),v,q,tight,origins,preserve)
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page
    for meta in e['real']:
        with PDFDocument((ROOT/meta['pdf']).read_bytes())as doc:
            p=prepare_table_page(doc[meta['page_index']]);tight=p.geometry.tight_bboxes;origins=p.geometry.origins
            v=e['cases'][meta['case_index']]['request']['table_input']
            for preserve in (False,True):
                recovered=recover(inp(v),tight,origins,preserve_font_styles=preserve)
                expected=None if recovered is None else dict(content=recovered[0],result=asdict(recovered[1]))
                real.append(dict(meta,request=dict(table_input=v,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve),expected=expected))
    prefix='rapid_doc/_vendor/docvortex_core/'
    files=['analyzers/native/pdf/table_text_styles.py','analyzers/native/pdf/table_materialization.py','analyzers/native/pdf/_table_recovery/engine.py','analyzers/native/pdf/_table_recovery/candidate.py','document/pdf/text/spacing.py']
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in files},renders=renders,recoveries=recoveries,real=real))
    print('Frozen',len(renders),'complete renders;',len(recoveries),'recoveries;',len(real),'real PDF/angle/style chains;',sum('<sup>'in c['expected']or'<sub>'in c['expected']for c in renders),'script-positive renders')

if __name__=='__main__':main()
