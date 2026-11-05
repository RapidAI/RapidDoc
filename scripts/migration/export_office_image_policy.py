"""Freeze Office placeholder and vector fallback policy, reference only."""
import gzip,json,sys,base64
from io import BytesIO
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office import image as original
    from loguru import logger
    logger.remove();cases=[]
    def pixels(data):
        with Image.open(BytesIO(data)) as image:return dict(size=list(image.size),rgba=list(image.convert('RGBA').tobytes()))
    for size in [(320,180),(1,1),(8,7),(40,30),(100,100),(301,179),(640,360)]:
        for lines in [[],[''],['WMF/EMF'],['hello','world'],list(original.STANDARD_VECTOR_PLACEHOLDER_LINES)]:
            image=original.create_text_placeholder(size,lines)
            cases.append(dict(size=list(size),lines=lines,rgba=list(image.convert('RGBA').tobytes())))
    source=original.get_standard_vector_placeholder_data_uri();placeholder=pixels(base64.b64decode(source.split(',')[1]))
    refs=ROOT/'tests/fixtures/office_rust_reference_068';f=json.loads(gzip.decompress((refs/'emfplus.json.gz').read_bytes()));vectors=[]
    for c in f['cases']:
        data=bytes.fromhex(c['hex']);source=original.serialize_office_image(data,part_name='vector.emf',content_type='image/emf')
        vectors.append(dict(name=c['name'],hex=c['hex'],placeholder=source.startswith('data:image/jpeg;')))
    for data in [b'',b'not vector',b'\xd7\xcd\xc6\x9a',b'\x01\0\x09\0']:
        source=original.serialize_office_image(data,part_name='vector.emf',content_type='image/emf')
        vectors.append(dict(name='bad '+data.hex(),hex=data.hex(),placeholder=source.startswith('data:image/jpeg;')))
    sizing=[]
    for size in [None,(0,20),(-1,3),(914400,457200),(2286,6858),(1,1),(9223372036854775807,2)]:
        sizing.append(dict(size=size,result=original._render_size_from_emu(size,dpi=200)))
    (refs/'office_image_policy.json.gz').write_bytes(gzip.compress(json.dumps(dict(placeholders=cases,standard=placeholder,vectors=vectors,sizing=sizing)).encode(),mtime=0))
    print('Office image references:',len(cases),'placeholders,',len(vectors),'vector policies,',len(sizing),'EMU sizes')
if __name__=='__main__':main()
