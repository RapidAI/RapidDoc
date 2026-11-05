"""Actual old raw-character -> glyph -> visual-lines -> 2D roles -> cell HTML chains."""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import FIELDS,clean

def main():
 path=ROOT/'build/migration/native-inline-cell-pipeline.json'
 if path.exists():raise FileExistsError(path)
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableResult,NativeTableCell
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text,build_cell_text_parts
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _cell_visual_lines,_cell_script_roles,_render_styled_cell,_cell_glyphs
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_merging import _merge_overlapping_inline_text_clusters
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
 cases=[]
 for a in (0,90,180,270):
  for height in (6.,12.,20.,30.,40.,80.):
   for preserve in (False,True):
    chars=[];p=(400.,600.);base='normal body A'
    for i,t in enumerate(base+'xy'):
     h=10. if i<len(base)else 6.;x=10.+i*5. if i<len(base)else 10.+len(base)*5.;y=60. if i<len(base)else 54. if t=='x'else 66.
     tight=[x,y-h,x+4.,y];loose=tight if t!='y'else[x,50.,x+4.,50.+height]
     def rot(b):return list(_rotate_bbox_from_upright(tuple(b),p,a))
     o=[x,y]if a==0 else[p[0]-y,x]if a==90 else[p[0]-x,p[1]-y]if a==180 else[y,p[1]-x]
     chars.append(dict(char=t,char_idx=i,bbox=rot(loose),tight_bbox=rot(tight),origin=o,font=dict(name='Arial-Bold'if preserve and i<2 else'Arial',flags=0,weight=700. if preserve and i<2 else 400.,size=h)))
    ti=NativeTableInput((0.,0.,400.,600.),p,a,tuple(chars));text=build_native_table_text(ti)
    assert text is not None
    by={q['char_idx']:q for q in chars};gs=sorted(text.glyphs,key=lambda g:(g.visual_row,g.bbox[0],g.bbox[1],g.glyph_id));content=''.join(t for t,source in build_cell_text_parts(gs,text.median_glyph_height));cell=NativeTableCell(0,0,1,1,(0.,0.,600. if a in(90,270)else 400.,400. if a in(90,270)else 600.),content,tuple(g.source_index for g in gs))
    result=NativeTableResult('<table/>',1,1,(cell,),text,'vector_grid',1.);gs=_cell_glyphs(result,cell);lines=_cell_visual_lines(gs,by,p,a);tight={q['char_idx']:q['tight_bbox']for q in chars};origins={q['char_idx']:q['origin']for q in chars}
    roles=_cell_script_roles(gs,by,p,a,tight,origins,set(),visual_lines=lines);merged=_merge_overlapping_inline_text_clusters(lines,p,[])
    html=_render_styled_cell(cell,gs,text.median_glyph_height,roles,chars_by_source=by,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve)
    cases.append(dict(label=f'{a}:{height}:{preserve}',text_request=dict(chars=chars,table_bbox=list(ti.table_bbox),angle=a),preparation_request=clean(dict(table_input=asdict(ti),result=asdict(result),tight_bboxes=tight)),expected_text=clean(asdict(text)),expected_merged_lines=clean([{k:getattr(l,k)for k in FIELDS}for l in merged]),origins=clean(origins),expected_roles=clean(roles),preserve_font_styles=preserve,expected_html=html,multiline=len(lines)>1,merge_positive=len(merged)<len(lines)))
 refs=['table_text_styles.py','line_merging.py','_table_recovery/text.py','native_text.py','inline/scripts.py']
 sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
 write_json(path,dict(sources=sources,cases=cases));print('Frozen',len(cases),'raw-char to cell HTML chains;',sum(c['multiline']for c in cases),'multi-line;',sum(c['merge_positive']for c in cases),'actual merges;',sum(bool(c['expected_roles'])for c in cases),'script-positive;',sum('<sup>'in c['expected_html']or'<sub>'in c['expected_html']for c in cases),'script HTML')
if __name__=='__main__':main()
