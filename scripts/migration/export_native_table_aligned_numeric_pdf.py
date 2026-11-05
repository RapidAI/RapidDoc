"""额外实际金额PDF→旧DocVortex原语→旧数值候选；固定文档，不覆写参考。"""
from io import BytesIO
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-aligned-numeric-pdf.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import aligned_numeric as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    cases=[];real=[]
    for kind,amounts in [('amounts',3),('status',2)]:
        path=ROOT/f'build/migration/aligned-numeric-{kind}.pdf'
        if path.exists():raise FileExistsError(path)
        data=BytesIO();canvas=Canvas(data,pagesize=(450,180),invariant=1);canvas.setFont('Helvetica',12)
        for y in [155,110,30]:canvas.line(10,y,440,y)
        canvas.drawString(20,125,'Item')
        for i in range(amounts):canvas.drawRightString(220+100*i,125,'Period '+chr(65+i))
        if kind=='status':canvas.drawString(350,125,'Status')
        for y,label,values in [(95,'Alpha',['1234','5678','9012']),(65,'Beta',['2345','6789','1234'])]:
            canvas.drawString(20,y,label)
            for i in range(amounts):canvas.drawRightString(220+100*i,y,values[i])
            if kind=='status':canvas.drawString(350,y,'OK')
        canvas.save();path.write_bytes(data.getvalue())
        with PDFDocument(path.read_bytes())as doc:
            p=prepare_table_page(doc[0])
            for angle in [0,90,180,270]:
                inp=NativeTableInput((10.,20.,440.,160.),p.page_size,angle,tuple(p.geometry.chars),p.drawing_lines,p.rectangles)
                text=build_native_table_text(inp)
                if text is None:continue
                regrouped=old.regroup_numeric_text(text);groups=old.numeric_row_groups(regrouped);candidate=old.build_aligned_numeric_candidate(inp,regrouped)
                cases.append(dict(request=dict(table_input=asdict(replace(inp,chars=())),text=asdict(text),regroup=True),expected=dict(text=asdict(regrouped),numeric_groups=groups,candidate=None if candidate is None else asdict(candidate),html=None if candidate is None else serialize_candidate_html(candidate))))
                real.append(dict(pdf=path.relative_to(ROOT).as_posix(),pdf_sha256=digest(path.read_bytes()),page_index=0,angle=angle,case_index=len(cases)-1))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(target,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in ['aligned_numeric.py','text.py','sparse_common.py']},cases=cases,real=real))
    print(f'Frozen {len(cases)} actual amount/status PDF inputs; {sum(c["expected"]["candidate"]is not None for c in cases)} nonempty candidates')
if __name__=='__main__':main()
