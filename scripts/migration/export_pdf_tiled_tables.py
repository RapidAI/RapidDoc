"""Actual PDF tiled candidates with an original generated test font, no result injection."""
import io,sys
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def font_bytes(tall):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    names=['.notdef','cjk','tall'];fb=FontBuilder(1000,isTTF=True);fb.setupGlyphOrder(names);fb.setupCharacterMap({0x4e2d:'cjk',88:'tall'})
    def glyph(h):
        p=TTGlyphPen(None);p.moveTo((0,0));p.lineTo((1000,0));p.lineTo((1000,h));p.lineTo((0,h));p.closePath();return p.glyph()
    fb.setupGlyf({'.notdef':glyph(1000),'cjk':glyph(1000),'tall':glyph(2800 if tall else 1000)});fb.setupHorizontalMetrics({n:(1000,0)for n in names});fb.setupHorizontalHeader(ascent=2800 if tall else 1000,descent=-200);name='TiledTallFixture'if tall else'TiledNormalFixture';fb.setupNameTable({'familyName':name,'styleName':'Regular','uniqueFontIdentifier':name,'fullName':name,'psName':name});fb.setupOS2(sTypoAscender=2800 if tall else 1000,sTypoDescender=-200,usWinAscent=2800 if tall else 1000,usWinDescent=200);fb.setupPost();fb.setupMaxp();out=io.BytesIO();fb.save(out);return out.getvalue()

def main():
    target=ROOT/'build/migration/pdf-tiled-tables.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_tiled_fill_table_candidates,has_inflated_native_cjk_metrics
    folder=ROOT/'build/migration/pdf-tiled-tables';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'tiled.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    for tall in [False,True]:pdfmetrics.registerFont(TTFont('Tiled'+str(tall),io.BytesIO(font_bytes(tall))))
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in ['plain','header','normal-metrics']:
            c.setPageRotation(rotation);c.setFillColorRGB(.7,.8,.9)
            for row in range(3):
                for col in range(2):c.rect(50+col*100,410+row*30,100,30,stroke=0,fill=1)
            c.setLineWidth(1);c.line(150,410,150,500)
            for y in [440,470]:c.line(50,y,250,y)
            c.setFillColorRGB(0,0,0);c.setFont('Tiled'+str(kind!='normal-metrics'),10);c.drawString(50,550,'中'*20);c.drawString(350,50,'X')
            for row in range(3):
                for col in range(2):c.drawString(65+col*100,420+row*30,'中中')
            if kind=='header':
                for x in [65,165]:c.drawString(x,510,'中中')
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,chars=g.chars,path_infos=[asdict(p)for p in snap.path_infos],drawing_lines=[asdict(r)for r in snap.drawing_lines],lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,chars=l.chars)for l in rows]))
            s=SimpleNamespace(page_size=snap.page_size,chars=g.chars,path_infos=snap.path_infos,drawing_lines=snap.drawing_lines,lines=rows)
            expected=clean(dict(candidates=[asdict(t)for t in _detect_tiled_fill_table_candidates(s,[])]))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=dict(source=source,excluded_bboxes=[]),expected=expected,inflated=has_inflated_native_cjk_metrics(g.chars)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'real PDF chains;',sum(bool(c['expected']['candidates'])for c in cases),'candidate positives;',sum(any(t['inferred_grid']for t in c['expected']['candidates'])for c in cases),'inferred-grid positives;',sum(c['inflated']for c in cases),'inflated positives')
if __name__=='__main__':main()
