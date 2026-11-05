"""Actual PDF detected filled candidates plus explicit duplicate/fragment queries to old merge."""
import json
from common import ROOT,digest,write_json
from export_native_table_candidate_merge import encode,reference
def main():
    target=ROOT/'build/migration/pdf-table-candidate-merge.json'
    if target.exists():raise FileExistsError(target)
    import export_pdf_table_replacements as prior
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_filled_grid_table_candidates,_detect_tiled_fill_table_candidates
    from dataclasses import asdict
    from types import SimpleNamespace
    from export_native_inline_clusters import clean
    corpus=json.loads((ROOT/'build/migration/pdf-table-replacements.json').read_text(encoding='utf-8'));cases=[]
    for c in corpus['cases']:
        data=(ROOT/c['pdf']).read_bytes();assert digest(data)==c['pdf_sha256']
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(c['page_index']);g=doc.get_page_chars_with_geometry(c['page_index']);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);s=SimpleNamespace(page_size=snap.page_size,chars=g.chars,path_infos=snap.path_infos,drawing_lines=snap.drawing_lines,lines=rows)
            detected=clean([asdict(t)for t in (_detect_filled_grid_table_candidates(snap.path_infos,snap.page_size,[])if c['kind']=='filled'else _detect_tiled_fill_table_candidates(s,[]))]);q=prior.queries(detected,c['mode']);raw=[encode(t)for k in ['base','recovered','replacements']for t in q[k]];cases.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],source=c['source'],kind=c['kind'],mode=c['mode'],request=dict(candidates=raw),expected=reference(raw)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Live filled detectors seed explicit duplicate/partial/halves candidate queries. Not full rule-draft materialization or general table production.',cases=cases));print('Frozen',len(cases),'live merge queries;',sum(c['request']!=c['expected']for c in cases),'changed')
if __name__=='__main__':main()
