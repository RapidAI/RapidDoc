"""Freeze original Office raster mode selection and decoded output pixels."""
import gzip,json,sys,random,struct,zlib
from pathlib import Path
from io import BytesIO
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def summary(source):
    if source is None:return None
    import base64
    mime,data=source.split(',',1)
    with Image.open(BytesIO(base64.b64decode(data))) as image:return dict(mime=mime.split(';')[0][5:],size=list(image.size),rgba=list(image.convert('RGBA').tobytes()))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.image import serialize_office_image
    from loguru import logger
    logger.remove();rng=random.Random(310314);cases=[]
    for size in [(3,2),(19,17)]:
        raw=Image.frombytes('RGBA',size,rng.randbytes(size[0]*size[1]*4))
        for mode in ['RGB','RGBA','L','LA','P','1','I;16','I','F','CMYK']:
            image=raw.convert(mode)
            for format in ['PNG','JPEG','BMP','GIF','TIFF','WEBP','PPM','TGA','ICO']:
                out=BytesIO()
                try:image.save(out,format=format)
                except (OSError,ValueError,KeyError):continue
                data=out.getvalue()
                try:expected=summary(serialize_office_image(data))
                except Exception as e:expected=dict(error=type(e).__name__)
                cases.append(dict(name=f'{size} {mode} {format}',hex=data.hex(),expected=expected))
        for mode in ['RGB','L','P','I;16']:
            image=raw.convert(mode);out=BytesIO()
            try:image.save(out,format='PNG',transparency=(20,30,40) if mode=='RGB' else 1)
            except (ValueError,OSError):continue
            data=out.getvalue();cases.append(dict(name=f'{size} {mode} PNG transparency',hex=data.hex(),expected=summary(serialize_office_image(data))))
    raw=Image.frombytes('RGBA',(7,5),rng.randbytes(7*5*4))
    for mode in ['RGB','RGBA','L','LA','P','1','I;16','CMYK']:
        image=raw.convert(mode)
        for compression in ['raw','tiff_lzw','tiff_adobe_deflate','packbits']+(['jpeg'] if mode in ['RGB','L','CMYK'] else []):
            out=BytesIO()
            try:image.save(out,format='TIFF',compression=compression)
            except (ValueError,OSError):continue
            data=out.getvalue();cases.append(dict(name=f'TIFF {mode} {compression}',hex=data.hex(),expected=summary(serialize_office_image(data))))
    for orientation in range(1,9):
        out=BytesIO();raw.convert('RGB').save(out,format='TIFF',tiffinfo={274:orientation});data=out.getvalue();cases.append(dict(name=f'TIFF orientation {orientation}',hex=data.hex(),expected=summary(serialize_office_image(data))))
    for transparency in [0,1,2]:
        out=BytesIO();raw.convert('P').save(out,format='GIF',transparency=transparency);data=out.getvalue();cases.append(dict(name=f'GIF transparent index {transparency}',hex=data.hex(),expected=summary(serialize_office_image(data))))
    for mode in ['RGB','L','CMYK']:
        bitmap=Image.frombytes('RGB',(19,35),rng.randbytes(19*35*3)).convert(mode);out=BytesIO();bitmap.save(out,format='TIFF',compression='jpeg',tiffinfo={278:16});data=out.getvalue();cases.append(dict(name='JPEG TIFF multiple strips '+mode,hex=data.hex(),expected=summary(serialize_office_image(data))))
    for mode in ['RGB','RGBA']:
        out=BytesIO();Image.new(mode,(9001,1),(20,30,40,80) if mode=='RGBA' else (20,30,40)).save(out,format='PNG');data=out.getvalue();cases.append(dict(name='ordinary image exceeds metafile width '+mode,hex=data.hex(),expected=summary(serialize_office_image(data))))
    for data in [b'',b'not image',b'\x89PNG\r\n\x1a\n',b'GIF89a',b'BM']:
        cases.append(dict(name='bad '+data.hex(),hex=data.hex(),expected=summary(serialize_office_image(data))))
    # Pillow validates header CRCs but skips IDAT CRC during load(), and may
    # complete a frame before an incomplete chunk tail. Preserve that policy.
    for mode in ['RGB','RGBA','P']:
        out=BytesIO();raw.convert(mode).save(out,format='PNG');base=out.getvalue()
        chunks=[];offset=8
        while offset<len(base):
            length=struct.unpack_from('>I',base,offset)[0];chunks.append((base[offset+4:offset+8],offset,length));offset+=length+12
        variants=[]
        huge=bytearray(base);struct.pack_into('>II',huge,16,2**31,2**31);variants.append(('bad header CRC huge dimensions',bytes(huge)))
        for name,p,n in chunks:
            changed=bytearray(base);changed[p+8+n]^=0x10;variants.append(('bad CRC '+name.decode(),bytes(changed)))
            if name==b'IDAT':
                variants.extend([('no IEND',base[:p+12+n]),('no IDAT CRC',base[:p+8+n]),('short IDAT payload',base[:p+8+n-3])])
                changed=bytearray(base);changed[p+4:p+8]=b'DDAT';variants.append(('DDAT first',bytes(changed)))
                changed=bytearray(base);changed[p+8+n-1]^=1;changed[p+8+n:p+12+n]=struct.pack('>I',zlib.crc32(changed[p+4:p+8+n]));variants.append(('bad Adler',bytes(changed)))
        for name,data in variants:
            try:expected=summary(serialize_office_image(data))
            except Exception as e:expected=dict(error=type(e).__name__)
            cases.append(dict(name='Pillow PNG '+mode+' '+name,hex=data.hex(),expected=expected))
    path=ROOT/'tests/fixtures/office_rust_reference_068/office_raster.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(cases=cases)).encode(),mtime=0));print('Office raster references:',len(cases))
if __name__=='__main__':main()
