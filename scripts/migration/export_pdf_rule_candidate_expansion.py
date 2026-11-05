"""Live PDF geometry to full ordinary rule candidate expansion; upstream projection explicit."""
import json
from common import ROOT,digest,write_json
from export_native_rule_annotation_preparation import row,fragment
from export_native_rule_candidate_expansion import reference
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
def main():
    target=ROOT/'build/migration/pdf-rule-candidate-expansion.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    cases=[];skipped=[]
    for name in ['pdf-table-footnotes','pdf-table-footnotes-closeaux']:
        corpus=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in corpus['cases']:
            data=(ROOT/c['pdf']).read_bytes();assert digest(data)==c['pdf_sha256']
            with PDFDocument(data)as doc:
                snap=doc._extract_native_page(c['page_index']);g=doc.get_page_chars_with_geometry(c['page_index']);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);frame=snap.path_infos[0].bbox;rows=[row([fragment(l.source_index,l.bbox,text=l.text)],visual=l.visual_row_id)for l in lines];core=[i for i,r in enumerate(rows)if frame[0]<=(r['bbox'][0]+r['bbox'][2])/2<=frame[2]and frame[1]<=r['center_y']<=frame[3]]
                if not core:skipped.append(c['label']);continue
                q=clean(dict(rows=rows,lines=[{k:getattr(l,k)for k in FIELDS}for l in lines],rule_bboxes=[r.bbox for r in snap.drawing_lines]or[frame],core_row_indices=core,page_size=snap.page_size,median_height=10.,angle=0,caption_source_index=next((l.source_index for l in lines if l.text.startswith('Table')),None),has_possible_table_notes=True));cases.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=name+'-'+c['label'],request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Live single-fragment visual-row projection; actual drawing boundary boxes or actual first fill-frame fallback, core rows by frame centres and fixture Table seed. No automatic rule-draft production claimed; notes selected automatically.',skipped_no_core=skipped,cases=cases));print('Frozen',len(cases),'live full candidate expansions;',sum(bool(c['expected']['candidate']['annotations'])for c in cases),'with annotations; skipped',len(skipped))
if __name__=='__main__':main()
