"""Freeze/verify installed RGB, RGBA and model-page pixels across a native update."""
import argparse
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    import rapid_doc_rust as native
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from accept_doc_performance import installed_binary
    source = ROOT / 'build/borrowed-render-strides-20261101.pdf'
    if not args.reference:
        canvas = Canvas(str(source), invariant=1)
        for width, rotation in [(31,0),(32,90),(33,180),(34,270),(1600,0)]:
            canvas.setPageSize((width, 50)); canvas.setPageRotation(rotation)
            canvas.setFillColorRGB(1,.2,.05); canvas.rect(0,0,width/2,50,fill=1,stroke=0)
            canvas.setFillColorRGB(.03,.4,.95); canvas.circle(width*.6,25,12,fill=1,stroke=0)
            canvas.showPage()
        canvas.save()
    cases = [(str(source.relative_to(ROOT)), i, 72.) for i in range(5)]
    cases += [('demo/pdfs/' + name + '.pdf', page, 144.) for name,page in [
        ('示例1-论文模板',0),('示例1-论文模板',1),('small_ocr',0),('49－长表格-密集型',0),('含印章',0)]]
    rows = []
    for path, page, dpi in cases:
        data = (ROOT/path).read_bytes()
        for mode in ['RGB','RGBA','model']:
            if mode == 'model':
                from rapid_doc_rust._native import render_pdf_model_page_bytes
                library, _ = native._prepare_pdf(data, None, None)
                width,height,png,scale = render_pdf_model_page_bytes(library,data,page,200.)
            else:
                rendered = native.render_pdf_page(data,page_index=page,dpi=dpi,color_mode=mode)
                width,height,png,scale = rendered.width,rendered.height,rendered.png,None
            with Image.open(io.BytesIO(png)) as image:
                pixels = image.tobytes(); image_mode = image.mode
            rows.append(dict(source=path,source_sha256=hashlib.sha256(data).hexdigest(),page=page,mode=mode,
                width=width,height=height,scale=scale,image_mode=image_mode,
                pixel_sha256=hashlib.sha256(pixels).hexdigest(),png_sha256=hashlib.sha256(png).hexdigest()))
    if args.reference:
        expected = json.loads(args.reference.read_text(encoding='utf8'))
        assert rows == expected['rows'], 'Rendered bytes changed; inspect saved input/reference before accepting'
    args.output.write_text(json.dumps(dict(installed_rust=installed_binary(),rows=rows,
        matches_reference=True if args.reference else None),ensure_ascii=False,indent=2),encoding='utf8')
    print(f'{len(rows)} rendered pages captured; exact={bool(args.reference)}')


if __name__ == '__main__':
    main()
