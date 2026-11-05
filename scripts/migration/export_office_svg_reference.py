"""Original resvg_py rasterization results, including sizing and rejection."""
import base64,gzip,hashlib,json,sys
from pathlib import Path
from io import BytesIO
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.foundation._svg_raster import serialize_svg_image
    from loguru import logger
    logger.remove();cases=[]
    def capture(name,data):
        try:
            result=serialize_svg_image(data)
            if result:
                with Image.open(BytesIO(base64.b64decode(result.split(',')[1]))) as image:expected=dict(size=list(image.size),sha256=hashlib.sha256(image.convert('RGBA').tobytes()).hexdigest())
            else:expected=None
        except Exception as e:expected=dict(error=type(e).__name__)
        cases.append(dict(name=name,hex=data.hex(),expected=expected))
    for w,h in [('24','16'),('1cm','2cm'),('100%','100%'),('',''),('100','100'),('9000','100'),('0','20'),('2in','1in')]:
        for view in ['', 'viewBox="0 0 30 20"']:
            for shape in ['<rect width="20" height="15" fill="#cd5315"/>','<circle cx="10" cy="8" r="7" fill="blue" fill-opacity=".4"/>','<path d="M1 1C20 30 4 20 20 3z" fill="green" stroke="red"/>']:
                text=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" {view}>{shape}</svg>';capture(f'size {w} {h} {view} {shape[:10]}',text.encode())
    for shape in ['','<text x="1" y="12">text without fonts</text>','<rect width="20" height="15" fill="none"/>','<script>alert(1)</script>','<defs><linearGradient id="g"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="20" height="15" fill="url(#g)"/>','<foreignObject width="20" height="15"><div xmlns="http://www.w3.org/1999/xhtml">text</div></foreignObject>']:
        capture(shape[:40],f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="16">{shape}</svg>'.encode())
    for encoding in ['utf-8','utf-16','windows-1252']:
        text=f'<?xml version="1.0" encoding="{encoding}"?><svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>';capture('encoding '+encoding,text.encode(encoding))
    for data in [b'bad svg',b'<svg',b'<html/>',b'<!DOCTYPE svg><svg/>',b'<!DOCTYPE svg [<!ENTITY a "hello">]><svg>&a;</svg>']:
        capture('bad '+data.hex(),data)
    path=ROOT/'tests/fixtures/office_rust_reference_068/office_svg.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(cases=cases)).encode(),mtime=0));print('Office SVG references:',len(cases))
if __name__=='__main__':main()
