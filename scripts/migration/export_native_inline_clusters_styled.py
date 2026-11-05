"""Positive typed multi-line role-to-HTML facts; supplied visual lines remain explicit stage inputs."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
 path=ROOT/'build/migration/native-inline-clusters-styled.json'
 if path.exists():raise FileExistsError(path)
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableCell,NativeTableGlyph
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_cell_text_parts
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.geometry import page_bbox_to_table_local
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _render_styled_cell
 f=json.loads((ROOT/'build/migration/native-inline-clusters.json').read_text());cases=[]
 for index,c in enumerate(f['cases']):
  if not c.get('expected_roles'):continue
  v=c['role_request'];glyphs=[];chars={}
  for row,line in enumerate(v['lines']):
   for q in line['chars']:
    id=q.get('char_idx');text=str(q.get('char')or'')
    if type(id)is not int:continue
    chars.setdefault(id,q)
    if not text or text.isspace():continue
    b=page_bbox_to_table_local(tuple(q['bbox']),(0.,0.,*v['page_size']),line['angle'])
    if b is not None:glyphs.append(NativeTableGlyph(len(glyphs),id,text,b,line.get('visual_row_id')if line.get('visual_row_id')is not None else row))
  glyphs.sort(key=lambda g:(g.visual_row,g.bbox[0],g.bbox[1],g.glyph_id));content=''.join(t for t,source in build_cell_text_parts(glyphs,10.));cell=NativeTableCell(0,0,1,1,(0.,0.,400.,600.),content,tuple(g.source_index for g in glyphs))
  roles={int(k):r for k,r in c['expected_roles'].items()};tight={int(k):b for k,b in v['tight_bboxes'].items()};origins={int(k):p for k,p in v['origins'].items()}
  for preserve in (False,True):
   request=clean(dict(cell=asdict(cell),glyphs=[asdict(g)for g in glyphs],median_height=10.,chars_by_source=chars,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve))
   html=_render_styled_cell(cell,glyphs,10.,roles,chars_by_source=chars,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve)
   cases.append(dict(case_index=index,label=f'{c["label"]}:{preserve}',request=request,expected=html))
 refs=['table_text_styles.py','_table_recovery/text.py','_table_recovery/geometry.py']
 sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
 write_json(path,dict(sources=sources,cases=cases));print('Frozen',len(cases),'positive typed line-role-to-HTML chains;',sum('<sup>'in c['expected']or'<sub>'in c['expected']for c in cases),'script HTML; visual lines supplied, not an end-to-end PDF claim')
if __name__=='__main__':main()
