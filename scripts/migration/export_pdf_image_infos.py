"""Freeze actual old PDF image-info reads, including authored pixel/geometry PDFs."""
import io,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/pdf-image-infos.json'
    if target.exists():raise FileExistsError(target)
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    folder=ROOT/'build/migration/pdf-image-infos';folder.mkdir(parents=True,exist_ok=True)
    generated=folder/'images.pdf'
    existing='--existing-pdf'in sys.argv
    if generated.exists()and not existing:raise FileExistsError(generated)
    if existing and not generated.is_file():raise FileNotFoundError(generated)
    canvas=Canvas(io.BytesIO()if existing else str(generated),pagesize=(400,600),pageCompression=0)
    labels=[]
    def draw(im,label,rotation=0,clip=None,form=False,full=True,flip=False):
        labels.append(label);canvas.setPageRotation(rotation)
        canvas.saveState()
        if clip:
            p=canvas.beginPath();p.rect(*clip);canvas.clipPath(p,stroke=0,fill=0)
        reader=ImageReader(im)
        if form:
            name='form'+str(len(labels));canvas.beginForm(name,0,0,400,600);canvas.drawImage(reader,0,0,400,600,mask='auto');canvas.endForm();canvas.translate(10,20);canvas.scale(.9,.9);canvas.doForm(name)
        else:
            if flip:canvas.translate(400,0);canvas.scale(-1,1)
            canvas.drawImage(reader,0,0 if full else 120,400 if full else 200,600 if full else 300,mask='auto')
        canvas.restoreState()
        canvas.setFont('Helvetica-Bold',18);canvas.drawString(25,575,'Display title')
        canvas.setFont('Helvetica',10)
        for y in [540,525,510]:canvas.drawString(25,y,'A sufficiently long body line for image background filtering.')
        canvas.showPage()
    for mode in ['L','RGB','RGBA']:
        for pattern in ['smooth','noise','blank-2','blank-3','blank-8','blank-12','blank-13','white']:
            im=Image.new(mode,(100,100),255 if mode=='L'else((255,255,255,255)if mode=='RGBA'else(255,255,255)))
            for y in range(100):
                for x in range(100):
                    if pattern=='smooth':v=180+x//5
                    elif pattern=='noise':v=(x*73+y*119)%256
                    elif pattern=='white':v=255
                    else:v=255 if y<int(pattern.split('-')[1])else 0
                    im.putpixel((x,y),v if mode=='L'else((v,v,v,255)if mode=='RGBA'else(v,v,v)))
            draw(im,mode+'-'+pattern)
            if pattern=='blank-8':
                for r in [90,180,270]:draw(im,mode+'-blank-rotation-'+str(r),rotation=r)
                draw(im,mode+'-blank-form',form=True)
                draw(im,mode+'-blank-top-clipped',clip=(0,0,400,550))
                draw(im,mode+'-blank-side-clipped',clip=(50,0,300,600))
                draw(im,mode+'-blank-small',full=False)
                draw(im,mode+'-blank-flip',flip=True)
    # Actual alpha must be ignored for smooth RGB but composited on white for blank-top.
    alpha=Image.new('RGBA',(100,100),(0,0,0,0))
    for y in range(8,100):
        for x in range(100):alpha.putpixel((x,y),(0,0,0,255))
    draw(alpha,'alpha-blank')
    for i in range(3):draw(Image.new('RGB',(80,80),(210,210,210)),'repeated-'+str(i))
    draw(Image.new('L',(1,300),240),'narrow-gray')
    canvas.save()
    paths={str(generated.relative_to(ROOT)):set(range(len(labels)))}
    regions=json.loads((ROOT/'build/migration/native-image-regions.json').read_text(encoding='utf-8'))
    for c in regions['cases']:
        for ref in c.get('pdf_pages',[]):paths.setdefault(ref['pdf'],set()).add(ref['page_index'])
    cases=[]
    for path,indices in paths.items():
        data=(ROOT/path).read_bytes()
        import pypdfium2 as pdfium
        with PDFDocument(data)as doc, pdfium.PdfDocument(data)as frames:
            for index in sorted(indices):
                expected=dict(page_size=doc.page_size(index),page_bbox=None,rotation=doc.page_rotation(index),image_infos=[asdict(i)for i in doc.get_page_image_infos(index)])
                # Frame comes from PDFium directly under the old document's process only.
                p=frames[index]
                try:expected['page_bbox']=p.get_bbox()
                finally:p.close()
                c=dict(pdf=path,pdf_sha256=digest(data),page_index=index,expected=clean(expected))
                if path==str(generated.relative_to(ROOT)):c['label']=labels[index]
                cases.append(c)
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/native_objects.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_coordinates.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_contracts.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'actual old PDF image pages;',sum(len(c['expected']['image_infos'])for c in cases),'images;',sum(i['smooth_background']for c in cases for i in c['expected']['image_infos']),'smooth;',sum(i['blank_top_bbox']is not None for c in cases for i in c['expected']['image_infos']),'blank tops')
if __name__=='__main__':main()
