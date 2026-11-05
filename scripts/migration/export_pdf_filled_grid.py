"""Freeze actual PDF paths to enclosing filled-row table candidates, four directions."""
import io,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_filled_grid import reference
def main():
    target=ROOT/'build/migration/pdf-filled-grid.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    folder=ROOT/'build/migration/pdf-filled-grid';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'filled.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in ['full','no-outer','three-rows','half-row-shading']:
            c.setPageRotation(rotation);c.setFillColorRGB(.8,.8,.9)
            if kind!='no-outer':c.rect(40,100,200,120,fill=1,stroke=0)
            n=3 if kind=='three-rows' else 4
            for r in range(n):
                for col in range(2):
                    c.setFillColorRGB(.2+col*.1,.4+r*.05,.6);c.rect(40+col*100,100+r*120/n,100,120/n,fill=1,stroke=0)
                    if kind=='half-row-shading':c.setFillColorRGB(.7,.7,.8);c.rect(40+col*100,100+r*120/n,100,60/n,fill=1,stroke=0)
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();cases=[]
    def extract(path,indices):
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            for i,label in indices:
                snap=doc._extract_native_page(i);paths=clean([asdict(p)for p in snap.path_infos]);q=dict(path_infos=paths,page_size=clean(snap.page_size),excluded_bboxes=[])
                cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=reference(paths,snap.page_size)))
    extract(pdf,list(enumerate(labels)))
    tiled=json.loads((ROOT/'build/migration/pdf-tiled-tables.json').read_text(encoding='utf-8'));path=ROOT/tiled['cases'][0]['pdf'];assert digest(path.read_bytes())==tiled['cases'][0]['pdf_sha256'];extract(path,[(c['page_index'],'tiled-'+c['label'])for c in tiled['cases']])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_constants.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual PDF filled-grid cases;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
