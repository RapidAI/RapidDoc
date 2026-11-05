"""Freeze actual old PNG materialization, with decoded-image equivalence explicit."""
import base64,copy,json,re
from io import BytesIO
from pathlib import Path
from PIL import Image
from common import ROOT,digest,write_json

URI=re.compile(r'data:image/png;base64,([A-Za-z0-9+/=]+)')
def projection(pages):
    pages=copy.deepcopy(pages);images=[]
    def replace(m):
        data=base64.b64decode(m[1])
        with Image.open(BytesIO(data))as image:
            image.load();record=dict(mode=image.mode,width=image.width,height=image.height,pixels_sha256=digest(image.tobytes()))
            if image.info.get('icc_profile'):record['icc_profile_sha256']=digest(image.info['icc_profile'])
            if 'transparency'in image.info:record['transparency']=image.info['transparency']
        images.append(record);return f'@PNG{len(images)-1}@'
    for page in pages:
        for block in page:
            if isinstance(block.get('content'),str):block['content']=URI.sub(replace,block['content'])
    return dict(pages=pages,images=images)

def main():
    target=ROOT/'build/migration/native-table-media-png.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_media import materialize_table_media as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _normalize_bbox_to_unit
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from loguru import logger
    logger.remove();folder=ROOT/'build/migration/table-media-png';folder.mkdir(parents=True,exist_ok=True)
    cases=[];pdf=[]
    def make(mode,seed):
        channels={'L':1,'LA':2,'RGB':3,'RGBA':4}[mode];w,h=37,29
        data=bytes((x*17+y*37+c*61+seed*13)%256 for y in range(h)for x in range(w)for c in range(channels))
        path=folder/f'{mode}-{seed}.png'
        if not path.exists():
            with Image.frombytes(mode,(w,h),data)as image:image.save(path)
        return path
    class PNGSource:
        def __init__(self,path):self.path=path;self.calls=[]
        def render_page(self,index,*,scale):
            assert scale==2.;self.calls.append(index)
            class Page:pass
            p=Page()
            with Image.open(self.path)as image:p.pil_image=image.copy()
            return p
    def normalized(pages,sizes):
        pages=copy.deepcopy(pages)
        if sizes:
            for i,page in enumerate(pages):
                for block in page:
                    if sizes[i]:
                        for item in block.get('_table_images')or[]:item['bbox']=_normalize_bbox_to_unit(tuple(item['bbox']),tuple(sizes[i]))
        return pages
    def add(label,pages,path,sizes=None):
        source=PNGSource(path);expected=normalized(pages,sizes);old(expected,source)
        cases.append(dict(label=label,blocks=pages[0],page_size=None if sizes is None else sizes[0],source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path.read_bytes()),expected_pages=expected,expected=projection(expected),render_calls=source.calls))
    def block(boxes,angle=0):
        return dict(type='table',bbox=[0.,0.,1.,1.],angle=angle,content='<table>'+''.join(f'<td><img src="docvortex-table-image:{i}" alt="Image" /></td>'for i in range(len(boxes)))+'</table>',_table_images=[dict(bbox=list(b),text=f'<&>"中文\' {i}')for i,b in enumerate(boxes)])
    valid=[.05,.07,.71,.83];reverse=[.8,.3,.2,.7];outside=[1.2,1.2,1.4,1.4];zero=[0.,0.,0.,0.]
    for n in range(160):
        mode=('L','LA','RGB','RGBA')[n%4];angle=(0,90,180,270)[(n//4)%4];kind=(n//16)%10;path=make(mode,n%3)
        boxes=[valid,[.2,.3,.6,.7]]
        if kind==1:boxes=[valid,reverse,valid]
        if kind==2:boxes=[outside,valid]
        if kind==3:boxes=[valid,zero]
        if kind==4:boxes=[[-.2,-.3,.2,.3],[.9,.9,1.1,1.1]]
        if kind==5:boxes=[valid]*13
        pages=[[block(boxes,angle),block([valid],270),dict(type='text',content='ordinary'),dict(type='table',content='empty metadata',_table_images=[])]]
        if kind==5:pages[0][0]['content']+='duplicate <img src="docvortex-table-image:0" alt="Image" /> and docvortex-table-image:10"'
        if kind==6:pages=[[dict(type='text',content='no media'),dict(type='table',content='empty',_table_images=[])]]
        if kind==7:
            path=folder/'invalid.png'
            if not path.exists():path.write_bytes(b'not PNG')
        sizes=None
        if kind in(8,9):
            size=[100.25,79.5]if kind==8 else[1000.,1000.];sizes=[size]
            for b in pages[0]:
                for i,item in enumerate(b.get('_table_images')or[]):item['bbox']=[.5+i,1.5,10.5+i,19.5]if kind==9 else[5.5,6.25,70.5,63.25]
        add(f'synthetic:{n}',pages,path,sizes)
    metadata=json.loads((ROOT/'build/migration/legacy-assets-metadata.json').read_text(encoding='utf-8'))
    for n,c in enumerate(metadata['cases']):
        for a in(90,270):add(f'metadata:{n}:{a}',[[block([valid],a)]],ROOT/c['source'])
    # Actual PDF images use the real old document; no rendering substitute.
    paths=[('tests/migration/inputs/pdf_native.pdf',0),('tests/migration/inputs/pdf_scan.pdf',0)]+[('tests/migration/inputs/pdf_mixed.pdf',i)for i in range(3)]+[(f'build/migration/geometry/forms-rot{a}.pdf',0)for a in(0,90,180,270)]
    large=folder/'large.pdf'
    if not large.exists():
        from reportlab.pdfgen import canvas
        c=canvas.Canvas(str(large),pagesize=(2400.25,800.5))
        for y in range(8):
            for x in range(24):c.setFillColorRGB(x/24,y/8,(x+y)%5/5);c.rect(x*100.,y*100.,95.,95.,fill=1,stroke=0)
        c.save()
    paths.append((large.relative_to(ROOT).as_posix(),0))
    class CountDocument:
        def __init__(self,doc):self.doc=doc;self.calls=[]
        def render_page(self,index,*,scale):self.calls.append(index);return self.doc.render_page(index,scale=scale)
    def add_pdf(label,path,pages,sizes=None):
        with PDFDocument((ROOT/path).read_bytes())as doc:
            source=CountDocument(doc);expected=normalized(pages,sizes);old(expected,source)
        pdf.append(dict(label=label,pdf=path,pdf_sha256=digest((ROOT/path).read_bytes()),pages=pages,page_sizes=sizes,expected_pages=expected,expected=projection(expected),render_calls=source.calls))
    for name,index in paths:
        for a in(0,90,180,270):
            pages=[[]for _ in range(index+1)];pages[index]=[block([valid,[.2,.2,.4,.4]],a),block([valid],180)]
            add_pdf(f'{name}:{index}:{a}',name,pages)
    add_pdf('nonexistent-second-page','tests/migration/inputs/pdf_native.pdf',[[block([valid])],[block([valid])]])
    full=json.loads((ROOT/'build/migration/native-table-block-commit.json').read_text(encoding='utf-8'))
    for n,c in enumerate([c for c in full['cases']if c['events'].get('media')][::6]):
        pages=[[],[],copy.deepcopy(c['expected']['table_blocks'])];sizes=[None,None,c['request']['source']['page_size']]
        add_pdf(f'commit-to-png:{n}','build/migration/model-document/five-pages.pdf',pages,sizes)
    prefix='rapid_doc/_vendor/docvortex_core/'
    files=['analyzers/native/pdf/table_media.py','analyzers/native/pdf/geometry.py','document/pdf/native_text_geometry.py','foundation/image_encoding.py']
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in files},encoding_equivalence='Rust PNG bytes/data URIs and URI-derived names may differ; decoded mode/size/pixels/ICC/transparency and all other JSON/HTML match',cases=cases,pdf=pdf))
    print('Frozen',len(cases),'PNG page cases;',len(pdf),'real PDF/angle/chain cases;',sum(len(c['expected']['images'])for c in cases),'synthetic PNG references;',sum(len(c['expected']['images'])for c in pdf),'real PDF PNG references')
if __name__=='__main__':main()
