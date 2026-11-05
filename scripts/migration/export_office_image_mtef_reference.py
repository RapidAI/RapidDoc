"""Original image equation extraction, cache, warnings and budget behavior."""
from pathlib import Path
import gzip,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
def main():
    import _mtef_test_utils as v3
    import _mtef_v5_test_utils as v5
    import _image_mtef_test_utils as image
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.equation.image import OfficeImageEquationDecoder
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeResourceLimitError
    from loguru import logger
    logger.remove();cases=[]
    def capture(payload,name,part=None,mime=None):
        decoder=OfficeImageEquationDecoder()
        case=dict(name=name,hex=payload.hex(),part=part,mime=mime)
        try:
            case['latex']=decoder.decode(payload,part_name=part,content_type=mime)
            assert decoder.decode(payload,part_name=part,content_type=mime)==case['latex']
        except LegacyOfficeResourceLimitError:case['error']='resource_limit'
        case.update(total_bytes=decoder.total_bytes,warnings=len(decoder._warned))
        cases.append(case)
    corpus=v3.formula_corpus()+v5.v5_formula_corpus()+v5.v5_template_corpus()
    for name,payload,_ in corpus:
        capture(image.build_gif_with_mtef(payload,chunk_size=3,include_baseline=True),name+' GIF')
        capture(image.build_wmf([image.pre6_wmf_comment(payload)]),name+' WMF pre6')
        capture(image.build_wmf(image.apps_mfcc_comments(payload,chunk_size=7),placeable=True),name+' WMF AppsMFCC')
    x=v3.mtef_equation(v3.mtef_char('x'));y=v5.v5_equation(v5.v5_char('y'))
    capture(image.build_baseline_only_gif(),'baseline GIF')
    capture(image.build_wmf([image.baseline_wmf_comment()]),'baseline WMF')
    capture(v3._TINY_PNG,'ordinary PNG')
    capture(v3._TINY_PNG,'incorrect GIF MIME',mime='image/gif')
    capture(v3._TINY_PNG,'dotfile hint',part='.gif')
    for payloads in [[x,x],[x,y],[x,b'invalid'],[b'invalid']]:
        capture(image.build_gif_with_extensions([image.gif_mtef_extension(p) for p in payloads]),'GIF candidates '+str(len(payloads))+' '+str(payloads[-1][:3]))
        capture(image.build_wmf([image.pre6_wmf_comment(p) for p in payloads]),'WMF candidates '+str(len(payloads))+' '+str(payloads[-1][:3]))
    for signature in ['Design Science, Inc./MTEF','WIRIS','Design Science','company/MTEF/product','company/other',' /MTEF','company\\/name/MTEF','bad\\','/MTEF','company//MTEF']:
        capture(image.build_wmf([image.apps_mfcc_comment(x,total_length=len(x),signature=signature)]),'signature '+signature)
    chunks=image.apps_mfcc_comments(x,chunk_size=3)
    for name,comments in [('missing tail',chunks[:-1]),('wrong order',list(reversed(chunks))),('restart foreign',[chunks[0],image.apps_mfcc_comment(y,total_length=len(y))]),('broken chunk',[chunks[0],b'AppsMFCC',*chunks[1:]]),('foreign chunk',[chunks[0],image.apps_mfcc_comment(y,total_length=len(y),signature='foreign/other'),*chunks[1:]])]:
        capture(image.build_wmf(comments),name)
    for name,payload in [('GIF',image.build_gif_with_mtef(x)),('WMF',image.build_wmf([image.pre6_wmf_comment(x)]))]:
        for size in [0,4,12,18,len(payload)//2,len(payload)-1]:capture(payload[:size],f'{name} truncated {size}',part='file.'+name.lower())
        capture(payload+b'garbage',name+' invalid tail')
        capture(payload+b'\x00\x00',name+' padding tail')
    deep=bytes([3,1,1,3,0])+b'\x01'*65+v3.mtef_char('x')+b'\x00'*66
    capture(image.build_gif_with_mtef(deep),'GIF deep resource error')
    capture(image.build_wmf([image.pre6_wmf_comment(deep)]),'WMF deep resource error')
    manifest=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/manifest.json').read_text())
    real=0
    for case in manifest['cases']:
        if case['format'] not in {'docx','xlsx','pptx'}:continue
        with zipfile.ZipFile(ROOT/case['input']) as package:
            for name in package.namelist():
                if '/media/' in name:
                    capture(package.read(name),case['id']+':'+name,part=name);real+=1
    target=ROOT/'tests/fixtures/office_rust_reference_068/image_mtef.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,real_images=real),ensure_ascii=True).encode(),mtime=0))
    print('Image MTEF references:',len(cases),'cases;',real,'real images;',sum(c.get('latex') is not None for c in cases),'decoded formulas')
if __name__=='__main__':main()
