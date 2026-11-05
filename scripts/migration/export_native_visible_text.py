"""Actual original visible-only extraction: clips, transparency, overpaint and resource vetoes."""
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-visible-text.json';pdf=ROOT/'build/migration/native-visible-text.pdf'
    if target.exists():raise FileExistsError('immutable visible text corpus exists')
    from reportlab.pdfgen.canvas import Canvas
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import NameObject,DictionaryObject,FloatObject,BooleanObject
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from loguru import logger
    logger.remove();buf=BytesIO();c=Canvas(buf,pagesize=(400,600),pageCompression=0,invariant=1);labels=[]
    modes=['visible','render-3','render-7','alpha-zero','stroke-visible','clip','empty-clip','form-clip','cover','partial-cover','earlier-cover','nested-cover','stroke-cover','transparent-cover','blend-veto','group-veto','inherited-veto','nonrect-cover','nonrect-clip']
    for rotation in [0,90,180,270]:
        for mode in modes:
            labels.append((rotation,mode));c.saveState()
            if mode=='earlier-cover':c.rect(20,490,350,90,stroke=0,fill=1)
            if mode in ['clip','empty-clip','form-clip']:
                p=c.beginPath();p.rect(60 if mode!='empty-clip'else 350,540,100 if mode!='empty-clip'else 10,25);c.clipPath(p,stroke=0)
            if mode=='nonrect-clip':
                p=c.beginPath();p.moveTo(20,490);p.lineTo(370,490);p.lineTo(180,590);p.close();c.clipPath(p,stroke=0)
            if mode=='alpha-zero':c.setFillAlpha(0)
            if mode=='form-clip':c.beginForm('clipped-form-'+str(rotation),0,0,400,600)
            t=c.beginText(30,550);t.setFont('Helvetica',12)
            if mode in ['render-3','render-7','stroke-visible']:t.setTextRenderMode({'render-3':3,'render-7':7,'stroke-visible':1}[mode])
            t.textOut('Original visible English words for clipping and painting');c.drawText(t)
            if mode=='form-clip':c.endForm();c.doForm('clipped-form-'+str(rotation))
            if mode in ['cover','partial-cover','nested-cover','stroke-cover','transparent-cover','blend-veto','group-veto','inherited-veto','nonrect-cover','nonrect-clip']:
                if mode=='nested-cover':c.beginForm('cover-form-'+str(rotation),0,0,400,600)
                if mode=='transparent-cover':c.setFillAlpha(.5)
                if mode=='nonrect-cover':
                    p=c.beginPath();p.moveTo(20,490);p.lineTo(370,490);p.lineTo(350,580);p.close();c.drawPath(p,stroke=0,fill=1)
                else:c.rect(20,490,100 if mode=='partial-cover'else 350,90,stroke=1 if mode=='stroke-cover'else 0,fill=0 if mode=='stroke-cover'else 1)
                if mode=='nested-cover':c.endForm();c.doForm('cover-form-'+str(rotation))
            c.restoreState();c.showPage()
    c.save();writer=PdfWriter();reader=PdfReader(BytesIO(buf.getvalue()))
    for page,(rotation,mode)in zip(reader.pages,labels):
        writer.add_page(page);page=writer.pages[-1];page.rotate(rotation)
        if mode in ['blend-veto','inherited-veto']:
            resources=page['/Resources'];resources[NameObject('/ExtGState')]=DictionaryObject({NameObject('/Unknown'):DictionaryObject({NameObject('/BM'):NameObject('/Multiply')})})
            if mode=='inherited-veto':page['/Parent'].get_object()[NameObject('/Resources')]=resources;page.pop('/Resources')
        if mode=='group-veto':page[NameObject('/Group')]=DictionaryObject({NameObject('/S'):NameObject('/Transparency')})
    generated=BytesIO();writer.write(generated)
    if pdf.exists():assert pdf.read_bytes()==generated.getvalue(),'immutable visibility PDF differs'
    else:pdf.write_bytes(generated.getvalue())
    cases=[]
    with PDFDocument(pdf.read_bytes())as doc:
        for i,(rotation,mode)in enumerate(labels):
            snapshot=doc._extract_native_page(i);g=snapshot.text_geometry
            assert g is not None,'freeze reference with the Python backend'
            cases.append(dict(label=f'{rotation}/{mode}',page_index=i,expected=dict(page_size=list(snapshot.page_size),chars=g.chars,tight_bboxes=g.tight_bboxes,loose_bboxes=g.loose_bboxes,origins=g.origins)))
    write_json(target,clean(dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=digest(pdf.read_bytes()),cases=cases)));print(len(cases),'real PDF visibility cases;',sum(not c['expected']['chars']for c in cases),'fully excluded pages')
if __name__=='__main__':main()
