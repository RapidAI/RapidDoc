"""Freeze empty/control character boundaries in paired geometry and compact line semantics."""
import copy,json
from common import ROOT,digest,write_json
from export_native_inline_clusters import FIELDS,clean

def main():
 target=ROOT/'build/migration/native-inline-clusters-empty.json'
 if target.exists():raise FileExistsError(target)
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _script_line_char_roles
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_merging import _merge_overlapping_inline_text_clusters
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _cell_script_roles
 f=json.loads((ROOT/'build/migration/native-inline-clusters.json').read_text(encoding='utf-8'))
 base=next(c['request']['lines']for c in f['cases']if c['label']=='paired:0');cases=[]
 for index,text in enumerate(['',' ','\n',None]):
  for position in ('before_sup','after_sub'):
   for geometry in (False,True):
    raw=copy.deepcopy(base);full=[q for l in raw for q in l['chars']];insert=len(full)-2 if position=='before_sup'else len(full)
    for q in full:
     if q['char_idx']>=insert:q['char_idx']+=1
    q=dict(char=text,char_idx=insert,bbox=[74.,50.,75.,60.])
    if geometry:q.update(tight_bbox=[74.,50.,75.,60.],origin=[74.,60.])
    raw[0]['chars'].append(q)
    chars=sorted([q for l in raw for q in l['chars']],key=lambda q:q['char_idx']);tight={q['char_idx']:q['tight_bbox']for q in chars if q.get('tight_bbox')is not None};origins={q['char_idx']:q['origin']for q in chars if q.get('origin')is not None}
    l=_LineItem(text='normal body Axy',bbox=(10.,46.,90.,70.),angle=0,source_index=0,chars=chars,compact_formula_cluster=True)
    cs,roles,counts,flags=_script_line_char_roles(l,(400.,600.),tight,origins,set())
    line=dict(text=l.text,bbox=list(l.bbox),angle=0,source_index=0,chars=chars,compact_formula_cluster=True)
    obj=[]
    for r in copy.deepcopy(raw):r['font_signature']=tuple(r['font_signature']);obj.append(_LineItem(**r))
    merged=_merge_overlapping_inline_text_clusters(obj,(400.,600.),[])
    roles_by_source=_cell_script_roles([],{},(400.,600.),0,tight,origins,set(),visual_lines=obj)
    cases.append(dict(label=f'{index}:{position}:{geometry}',request=clean(dict(lines=raw,page_size=[400.,600.],table_bboxes=[])),expected_lines=clean([{k:getattr(l,k)for k in FIELDS}for l in merged]),role_request=clean(dict(lines=raw,page_size=[400.,600.],tight_bboxes=tight,origins=origins,fraction_members=[])),expected_roles=clean(roles_by_source),line_request=clean(dict(line=line,page_size=[400.,600.],tight_bboxes=tight,origins=origins)),expected_line=clean(dict(chars=cs,roles=roles,body_counts=counts,formula_flags=flags))))
 refs=['inline/scripts.py','_script_geometry.py','line_merging.py','table_text_styles.py']
 sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
 write_json(target,dict(sources=sources,cases=cases));print('Frozen',len(cases),'empty/control compact-line and cluster boundary references')
if __name__=='__main__':main()
