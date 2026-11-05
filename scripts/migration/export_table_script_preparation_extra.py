"""Additional old typography limits and genuine one-line cell-to-HTML chains."""
import copy,json
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode
from export_table_script_preparation import METRICS

def main():
    target=ROOT/'build/migration/table-script-preparation-extra.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _fill_native_typography
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _cell_script_roles,_render_styled_cell
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph,NativeTableCell
    metrics=[];chains=[]
    shapes=[None,[],[0,0,0,2],[20,20,0,0],[-20,-10,-5,0],[95,190,110,210],['1','2','7','12']]
    for n in range(112):
        raw=[None,False,True,0,1,'',' ','\n','X','α'][n%10];bbox=shapes[n%len(shapes)]
        cs=[dict(char=raw,bbox=bbox,font=dict(name=['Arial','',None][n%3],flags=['bad',None,1.9,'+64','٦٤',2**64-1,-2**63][n%7],weight=None))]
        v=dict(line=dict(text='.',bbox=[0.,0.,100.,200.],angle=(0,90,180,270)[n%4],chars=cs,em_height=0.),page_size=[100.,200.])
        l=_LineItem(text='.',bbox=tuple(v['line']['bbox']),angle=v['line']['angle'],source_index=0,chars=cs)
        _fill_native_typography(l,tuple(v['page_size']));metrics.append(dict(label=f'edge:{n}',request=encode(v),expected=encode({k:getattr(l,k)for k in METRICS})))
    f=json.loads((ROOT/'build/migration/table-script-preparation.json').read_text(encoding='utf-8'))
    for n,c in enumerate(f['preparation']):
        v=c['request'];ti=v['table_input'];tight={int(k):q for k,q in v['tight_bboxes'].items()};origins={q['char_idx']:q.get('origin',(q['bbox'][0],q['bbox'][3]))for q in ti['chars']if type(q.get('char_idx'))is int}
        by={}
        for i,q in enumerate(ti['chars']):
            id=q.get('char_idx',i)
            if type(id)is int:by.setdefault(id,q)
        for ci,cell in enumerate(c['expected']['cells']):
            if len(cell['visual_lines'])!=1:continue
            line=cell['visual_lines'][0];l=_LineItem(text=line['text'],bbox=tuple(line['bbox']),angle=line['angle'],source_index=line['source_index'],visual_row_id=line['visual_row_id'],chars=line['chars'])
            gs=[NativeTableGlyph(**g)for g in cell['glyphs']]
            roles=_cell_script_roles(gs,by,tuple(ti['page_size']),ti['angle'],tight,origins,set(cell['fraction_members']),visual_lines=[l])
            source=next(q for q in v['result']['cells']if(q['row'],q['col'])==(cell['row'],cell['col']))
            for preserve in (False,True):
                html=_render_styled_cell(NativeTableCell(**source),gs,v['result']['text']['median_glyph_height'],roles,chars_by_source=by,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve)
                chains.append(dict(label=f'chain:{n}:{ci}:{preserve}',preparation_index=n,cell_index=ci,origins=encode(origins),preserve_font_styles=preserve,expected_roles=encode(roles),expected_html=html))
    refs=['native_text.py','typography.py','table_text_styles.py','geometry.py','inline/scripts.py','line_merging.py']
    sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
    write_json(target,dict(sources=sources,metrics=metrics,chains=chains));print('Frozen',len(metrics),'edge metrics and',len(chains),'one-line preparation/classification/render chains')
if __name__=='__main__':main()
