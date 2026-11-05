"""Freeze original v3/v5 MTEF behavior and native/OLE envelopes."""
from pathlib import Path
import gzip,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
def main():
    import _mtef_test_utils as v3
    import _mtef_v5_test_utils as v5
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.equation.mtef import decode_mtef,decode_equation_native,decode_equation_object,read_object_pool_equations
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeResourceLimitError
    from loguru import logger
    logger.remove();cases=[]
    def capture(data,name,kind='mtef'):
        case=dict(name=name,kind=kind,hex=data.hex())
        try:
            if kind=='pool':
                with BoundedOleReader(data) as ole:case['latex']=read_object_pool_equations(ole)
            else:case['latex']={'mtef':decode_mtef,'native':decode_equation_native,'object':decode_equation_object}[kind](data)
        except LegacyOfficeResourceLimitError:case['error']='resource_limit'
        cases.append(case)
    corpus=v3.formula_corpus()+v5.v5_formula_corpus()+v5.v5_template_corpus()
    for name,data,expected in corpus:
        assert decode_mtef(data)==expected,name
        capture(data,name)
        capture(v3.equation_native(data),name+' native','native')
        blob=v3._build_nested_cfb([dict(name='Root Entry',kind=5,child=1),dict(name='Equation Native',kind=2,data=v3.equation_native(data))])
        capture(blob,name+' OLE','object')
    for selector in range(50):
        for variation in [0,1,2,3,255]:
            for slots in [[],[v3.mtef_line(v3.mtef_char(c)) for c in 'abcd']]:
                capture(v3.mtef_equation(v3.mtef_template(selector,*slots,variation=variation)),f'v3 selector {selector}/{variation}/{len(slots)}')
    for selector in range(39):
        for variation in [0,1,2,3,4,6,7,8,15,16,30,63,0x100,0x1ff,0x200]:
            for options in [0,1,2]:
                capture(v5.v5_equation(v5.v5_template(selector,*[v5.v5_line(v5.v5_char(c)) for c in 'abcd'],variation=variation,options=options)),f'v5 selector {selector}/{variation}/{options}')
    for embellishment in range(40):
        capture(v3.mtef_equation(v3.mtef_char('x',embellishments=(embellishment,))),f'v3 embellishment {embellishment}')
        capture(v5.v5_equation(v5.v5_char('x',embellishments=(embellishment,))),f'v5 embellishment {embellishment}')
    for face in [1,2,3,4,5,6,11,12,22,23,24,0,13,-1]:
        for code in [32,65,97,0x81,0x391,0xe930,0xf001,0xf01b,0xf080,0xf100,0xe111,0xd800]:
            capture(v5.v5_equation(v5.v5_char(chr(code),typeface=face)),f'v5 code {code:04x}/{face}')
            capture(v5.v5_equation(v5.v5_char('x',typeface=face,omit_mtcode=True,font_position=code)),f'v5 font position {code:04x}/{face}')
    for bits in range(5):
        definitions=v5.v5_font_definition(3,'Symbol')+v5.v5_font_style_definition(1,bits)
        capture(v5.v5_equation(definitions+v5.v5_char('x',typeface=-1)),f'font style {bits}')
    for name,data,_ in corpus[:8]:
        for size in sorted({0,1,4,5,6,7,len(data)//2,len(data)-1,len(data)}):capture(data[:size],f'truncated {name} {size}')
        capture(data+b'garbage',name+' trailing')
    for depth in [63,64,65]:
        capture(bytes([3,1,1,3,0])+b'\x01'*depth+v3.mtef_char('x')+b'\x00'*(depth+1),f'v3 depth {depth}')
        capture(bytes([5,1,1,6,0])+b'DSMT4\x00\x00'+b'\x01\x00'*depth+v5.v5_char('x')+b'\x00'*(depth+1),f'v5 depth {depth}')
    pool=v3._build_nested_cfb([dict(name='Root Entry',kind=5,child=1),dict(name='ObjectPool',kind=1,child=2),
        dict(name='_001',kind=1,child=3,right=4),dict(name='Equation Native',kind=2,data=v3.equation_native(v3.mtef_equation(v3.mtef_char('x')))),
        dict(name='_1',kind=1,child=5),dict(name='Equation Native',kind=2,data=v3.equation_native(v3.mtef_equation(v3.mtef_char('y'))))])
    capture(pool,'ObjectPool numeric aliases keep first','pool')
    # Real native streams and embedded Equation OLE objects from existing cases.
    manifest=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/manifest.json').read_text())
    real=0
    for case in manifest['cases']:
        path=ROOT/case['input'];data=path.read_bytes()
        if case['format'] in {'doc','xls','ppt'}:
            with BoundedOleReader(data) as ole:
                for name in ole.stream_names():
                    if name.casefold().endswith('equation native'):
                        capture(ole.read_stream(name),case['id']+':'+name,'native');real+=1
        elif case['format'] in {'docx','xlsx','pptx'}:
            with zipfile.ZipFile(path) as package:
                for name in package.namelist():
                    if '/embeddings/' in name and name.endswith('.bin'):
                        payload=package.read(name)
                        if payload.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                            capture(payload,case['id']+':'+name,'object');real+=1
    target=ROOT/'tests/fixtures/office_rust_reference_068/mtef.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,real_objects=real),ensure_ascii=True).encode(),mtime=0))
    print('MTEF references:',len(cases),'cases;',real,'real streams/objects;',sum('error' in c for c in cases),'resource errors')
if __name__=='__main__':main()
