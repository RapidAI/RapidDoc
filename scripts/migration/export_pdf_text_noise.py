"""Freeze actual fill-only paint and complete live character/source/noise decisions."""
import io,json
from dataclasses import fields
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-text-noise.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_text import restore_drop_cap_paragraphs,group_hanging_letter_paragraphs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_noise import _nested_prose_numeric_outliers_python as candidates,exclude_faint_native_noise as exclude
    folder=ROOT/'build/migration/pdf-text-noise';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'noise.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    canvas=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);specs=[]
    for gray in [169,170,200,225,226]:
        for mode in [0,1,2,3,4,7]:specs.append((gray,mode,0))
    specs.extend([(200,0,255),(200,0,121)])
    for gray,mode,body in specs:
        canvas.setFillColorRGB(body/255,body/255,body/255);canvas.setFont('Helvetica',10)
        canvas.drawString(20,560,'alpha bravo charlie delta echo')
        canvas.drawString(20,540,'foxtrot golf hotel india juliet')
        canvas.setFillColorRGB(gray/255,gray/255,gray/255);canvas.setStrokeColorRGB(gray/255,gray/255,gray/255)
        text=canvas.beginText(60,550);text.setFont('Courier',8);text.setTextRenderMode(mode);text.textOut('12');canvas.drawText(text);canvas.showPage()
    canvas.save();data=pdf.read_bytes();names=[f.name for f in fields(_LineItem)if not f.name.startswith('_')];cases=[]
    with PDFDocument(data)as doc:
        for i,spec in enumerate(specs):
            geometry=doc.get_page_chars_with_geometry(i);size=doc.page_size(i);rotation=doc.page_rotation(i)
            lines=_build_native_line_items_from_chars(geometry.chars,size,page_rotation=rotation);restore_drop_cap_paragraphs(lines);group_hanging_letter_paragraphs(lines)
            rows=[{n:clean(getattr(l,n))for n in names}for l in lines];before=dict(lines=rows,chars=clean(geometry.chars),page_index=i)
            old=SimpleNamespace(lines=lines,chars=list(geometry.chars));positions={id(l):j for j,l in enumerate(lines)}
            cs=candidates(old,());requested=sorted({c['char_idx']for l,refs in cs for p in [l,*refs]for c in p.chars if c.get('char','').strip()and'char_idx'in c})
            prepared=dict(candidates=[dict(line_index=positions[id(l)],reference_indices=[positions[id(r)]for r in refs])for l,refs in cs],requested_char_indices=requested)
            # Requests include negative/out-of-range/duplicate indices, all render modes, and generated spaces.
            indices=[-1,*range(100),0,9999];paint=doc._get_page_text_paint(i,indices);calls=[]
            def reader(requested):calls.append(requested);return doc._get_page_text_paint(i,requested)
            removed=exclude(old,reader,());after=dict(lines=[rows[positions[id(l)]]for l in old.lines],chars=clean(old.chars),page_index=i)
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=str(spec),indices=indices,paint={str(k):list(v)for k,v in paint.items()},source=before,prepared=prepared,expected=dict(source=after,excluded_source_indices=sorted(removed),requested_char_indices=calls[0]if calls else[])))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/_document.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/text_noise.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=clean(cases)))
    print('Frozen',len(cases),'actual paint/source pages;',sum(bool(c['prepared']['candidates'])for c in cases),'candidate-positive;',sum(bool(c['expected']['excluded_source_indices'])for c in cases),'removal-positive')
if __name__=='__main__':main()
