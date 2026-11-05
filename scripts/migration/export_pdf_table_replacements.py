"""Live PDF extraction and old filled detectors to explicit replacement queries."""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_table_replacements import reference
def queries(candidates,mode):
    base=copy.deepcopy(candidates);new=copy.deepcopy(candidates)
    if mode!='self':
        new=[]
        for c in candidates:
            b=c.get('core_bbox')or c['bbox'];mid=(b[1]+b[3])/2
            for bounds in ([b[0],b[1],b[2],mid],)if mode=='partial'else([b[0],b[1],b[2],mid],[b[0],mid,b[2],b[3]]):
                c=copy.deepcopy(c);c['bbox']=c['local_bbox']=c['core_bbox']=bounds;new.append(c)
    return dict(base=base,recovered=[],replacements=new)
def main():
    target=ROOT/'build/migration/pdf-table-replacements.json'
    if target.exists():raise FileExistsError(target)
    from types import SimpleNamespace
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_filled_grid_table_candidates,_detect_tiled_fill_table_candidates
    filled=json.loads((ROOT/'build/migration/pdf-filled-grid.json').read_text(encoding='utf-8'));tiled=json.loads((ROOT/'build/migration/pdf-tiled-tables.json').read_text(encoding='utf-8'));inputs=[('filled',c)for c in filled['cases'][:16]]+[('tiled',c)for c in tiled['cases']];cases=[]
    for kind,c in inputs:
        path=ROOT/c['pdf'];data=path.read_bytes();assert digest(data)==c['pdf_sha256']
        with PDFDocument(data)as doc:
            i=c['page_index'];snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);s=SimpleNamespace(page_size=snap.page_size,chars=g.chars,path_infos=snap.path_infos,drawing_lines=snap.drawing_lines,lines=rows)
            candidates=clean([asdict(x)for x in (_detect_filled_grid_table_candidates(snap.path_infos,snap.page_size,[])if kind=='filled'else _detect_tiled_fill_table_candidates(s,[]))]);source=clean(dict(page_size=snap.page_size,chars=g.chars,path_infos=[asdict(x)for x in snap.path_infos],drawing_lines=[asdict(x)for x in snap.drawing_lines],lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,chars=l.chars)for l in rows]))
            for mode in (['self','partial','halves']if candidates else['self']):
                q=queries(candidates,mode);cases.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=i,label=kind+'-'+c['label']+'-'+mode,kind=kind,mode=mode,source=source,request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual detected filled candidates seed explicit self/partial/halves replacement queries; modifications are test queries, not full numeric/year/metric producers.',cases=cases));print('Frozen',len(cases),'live replacement queries across',len(inputs),'pages;',sum(bool(c['request']['base'])for c in cases),'nonempty;',sum(c['request']!=c['expected']for c in cases),'changed')
if __name__=='__main__':main()
