"""Freeze actual PDF compound/single bar geometry and zero-axis outputs, explicit upstream tables."""
import io,json,sys
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_geometry import reference
def main():
    target=ROOT/'build/migration/pdf-bar-geometry.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_union_many
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_tiled_fill_table_candidates
    prepare,axes,_=reference();folder=ROOT/'build/migration/pdf-bar-geometry';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'bars.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for compound in [True,False]:
            canvas.setPageRotation(rotation);canvas.setFillColorRGB(.2,.4,.7)
            for horizontal in [True,False]:
                p=canvas.beginPath()
                for i in range(4):
                    b=(50,140+i*20,50+i*20,10)if horizontal else(230+i*20,140,10,50+i*20)
                    if compound:p.rect(*b)
                    else:canvas.rect(*b,stroke=0,fill=1)
                if compound:canvas.drawPath(p,stroke=0,fill=1)
            canvas.setLineWidth(2);canvas.line(50,135,50,215);canvas.line(225,140,305,140);canvas.showPage();labels.append(f'{rotation}-{compound}')
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);geometry=doc.get_page_chars_with_geometry(i);original=list(snap.path_infos);rules=list(snap.drawing_lines);full=SimpleNamespace(page_size=snap.page_size,chars=geometry.chars,path_infos=original,lines=[],drawing_lines=rules)
            tables=[c.bbox for c in _detect_tiled_fill_table_candidates(full,[])];assert not tables
            flags=[];admitted=[];directional=[[],[]]
            for j,path in enumerate(original):
                s=SimpleNamespace(path_infos=[path],drawing_lines=rules,compound_baseline_cache={});accepted,directions,baseline=prepare(s,10.,tables);flags.append(baseline.get(id(path.rectangle_bboxes)))
                for bar in accepted:
                    ri=next((k for k,b in enumerate(path.rectangle_bboxes)if b is bar.bbox),None)if len(path.rectangle_bboxes)>1 else None
                    h=any(bar is b for b in directions[0]);v=any(bar is b for b in directions[1]);admitted.append(dict(path_index=j,rectangle_index=ri,bbox=bar.bbox,horizontal=h,vertical=v))
                    if h:directional[0].append(bar.bbox)
                    if v:directional[1].append(bar.bbox)
            queries=[]
            for h,boxes in zip([True,False],directional):
                if boxes:
                    b=_bbox_union_many(boxes)
                    for edge in ((0,2)if h else(3,1)):queries.append(dict(bounds=b,horizontal=h,baseline=boxes[0][edge]))
            expected=clean(dict(compound_baselines=flags,admitted_bars=admitted,zero_axis_bboxes=[axes(SimpleNamespace(drawing_lines=rules),q['bounds'],q['horizontal'],q['baseline'],10.)for q in queries]))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=clean(dict(path_infos=[asdict(p)for p in original],drawing_lines=[asdict(r)for r in rules],em=10.,table_bboxes=tables,axis_queries=queries)),expected=expected))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_graphic_geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual PDF bar preparation cases;',sum(bool(c['expected']['admitted_bars'])for c in cases),'admission-positive;',sum(any(c['expected']['zero_axis_bboxes'])for c in cases),'axis-positive')
if __name__=='__main__':main()
