"""Freeze focused original-Python references for borrowed/owned native stages."""
import copy
import gzip
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from loguru import logger
    logger.remove()
    from export_native_text_roles import scenes as roles, reference as role_reference
    from export_native_footnotes import scenes as notes
    from export_native_geometry import lines
    from export_native_late_image_text import project
    from export_native_inline_clusters import clean
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_merging import merge_text_line_clusters
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import detect_pdf_text_script_lines
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    target=ROOT/'tests/fixtures/native_owned_line_reference.json.gz'
    if target.exists():raise FileExistsError(target)
    clusters=[]
    for c in notes()[:32]:
        q=c['request']
        expected=[project(l) for l in merge_text_line_clusters(lines(copy.deepcopy(q['lines'])),tuple(q['page_size']),[tuple(b) for b in q['table_bboxes']])]
        clusters.append(dict(label=c['label'],request=q,expected=expected))
    pdf=ROOT/'build/native-owned-script-reference.pdf'
    canvas=Canvas(str(pdf),pagesize=(260,190),invariant=1)
    for rotation in [0,90,180,270]:
        canvas.setPageRotation(rotation);canvas.setFont('Helvetica',12);canvas.drawString(20,140,'Energy mc')
        canvas.setFont('Helvetica',7);canvas.drawString(80,145,'2')
        canvas.setFont('Helvetica',12);canvas.drawString(20,100,'Water H')
        canvas.setFont('Helvetica',7);canvas.drawString(64,96,'2')
        canvas.setFont('Helvetica',12);canvas.drawString(70,100,'O is liquid');canvas.showPage()
    canvas.save();scripts=[]
    with PDFDocument(pdf.read_bytes()) as doc:
        for index in range(4):
            g=doc.get_page_chars_with_geometry(index)
            with doc._open_page(index) as page:size=tuple(page.get_size());rotation=page.get_rotation()
            ls=_build_native_line_items_from_chars(g.chars,size,page_rotation=rotation,supported_angles=(0.,90.,180.,270.))
            result=detect_pdf_text_script_lines(ls,size,g.tight_bboxes,g.origins,all_chars=g.chars,drawing_lines=[])
            scripts.append(dict(label=str(rotation),request=dict(lines=[project(l) for l in ls],page_size=size,
                tight_bboxes=g.tight_bboxes,origins=g.origins,all_chars=g.chars,drawing_lines=[]),expected=[asdict(p) for p in result]))
    role_cases=roles()
    for case in role_cases:
        try:case['expected']=role_reference(case['request'])
        except ValueError as error:case.update(error='invalid_input',old_exception=type(error).__name__)
    sources=['analyzers/native/pdf/line_merging.py','analyzers/native/pdf/auxiliary_text.py',
             'analyzers/native/pdf/inline/scripts.py','analyzers/native/pdf/title_analysis/native_boundaries.py',
             'analyzers/native/pdf/title_analysis/structural.py','analyzers/native/pdf/text_assembly/continuity.py']
    hashes={name:hashlib.sha256((ROOT/'rapid_doc/_vendor/docvortex_core'/name).read_bytes()).hexdigest() for name in sources}
    target.write_bytes(gzip.compress(json.dumps(clean(dict(source_sha256=hashes,roles=role_cases,notes=notes(),clusters=clusters,scripts=scripts)),ensure_ascii=False).encode('utf8'),mtime=0))
    print('Owned line references:',len(role_cases),len(notes()),len(clusters),len(scripts))


if __name__=='__main__':main()
