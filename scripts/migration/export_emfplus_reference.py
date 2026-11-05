"""Freeze metafile-render 0.3.0 EMF+ replay, errors and output (reference only)."""
import dataclasses, gzip, hashlib, json, struct
from io import BytesIO
from pathlib import Path
from PIL import Image
from export_office_metafile_reference import emf, erec, plus, comment, u, i, h
from export_office_path_pixels import normalized_svg
ROOT = Path(__file__).resolve().parents[2]
V = 0xdbc01001
def f(*v): return struct.pack('<'+'f'*len(v), *v)
def obj(v):
    if dataclasses.is_dataclass(v): return obj(dataclasses.asdict(v))
    if isinstance(v, dict):
        if set(v)=={'png'}:
            with Image.open(BytesIO(v['png'])) as image:
                return dict(size=list(image.size), rgba=list(image.convert('RGBA').tobytes()))
        return {k:obj(x) for k,x in v.items()}
    if isinstance(v,(tuple,list,bytes)): return [obj(x) for x in v]
    return v
def define(kind,data,slot=0): return plus(0x4008,data,(kind<<8)|slot)
def brush(color=0x8060a020): return u(V,0,color)
def fill(): return plus(0x400a,u(0xff902030,1)+f(2,3,15,10),0x8000)
def font(name='Arial',style=0,unit=2):
    return u(V)+f(8)+u(unit,style,0,len(name))+name.encode('utf-16le')
def fmt(flags=0,align=0,vertical=0):
    return u(V,flags,0,align,vertical)+bytes(12)+u(0)+f(0,0,1)+u(0,0,0)
def image(payload,encoding=1,w=0,h=0,stride=0,pixel=0):
    return u(V,1)+i(w,h,stride)+u(pixel,encoding)+payload
def draw_image(points=False,flags=0):
    return plus(0x401b if points else 0x401a,u(0xffffffff,2)+f(0,0,3,2)+(u(3)+f(2,2,20,3,4,16) if points else f(2,2,18,14)),flags)
def main():
    from metafile_render.parser import parse_metafile
    from metafile_render.context import ReplayContext
    from metafile_render import render_metafile
    cases=[]
    def capture(name,records,wrapped=True):
        records=([plus(0x4001,u(V,0,96,96))]+records+[plus(0x4002)]) if wrapped else records
        data=emf([comment(records)])
        capture_file(name,data)
    def capture_file(name,data):
        c=dict(name=name,hex=data.hex())
        context=ReplayContext()
        try:
            c['document']=obj(parse_metafile(data,dpi=200,size_hint=(24,20),context=context))
            c['budget']=obj(context.budget)
        except Exception as e: c['error']=getattr(e,'code','malformed');cases.append(c);return
        c['outputs']={}
        for output in ['png','svg']:
            try:
                result=render_metafile(data,dpi=200,size_hint=(24,20),output_format=output,backend='replay')
                out=dict(width=result.width,height=result.height,partial=result.partial,diagnostics=obj(result.diagnostics))
                if output=='svg':out['svg']=normalized_svg(result.data)
                else:
                    with Image.open(BytesIO(result.data)) as image:out['rgba']=list(image.convert('RGBA').tobytes())
                c['outputs'][output]=out
            except Exception as e:c['outputs'][output]=dict(error=getattr(e,'code','malformed'))
        cases.append(c)
    capture('empty',[])
    capture('clear',[plus(0x4009,u(0x40305070))])
    capture('clear after fill',[fill(),plus(0x4009,u(0x40305070))])
    capture('clipped clear',[fill(),plus(0x4032,f(4,5,8,8)),plus(0x4009,u(0x10305070))])
    for compressed in [0,0x4000]:
        rect=h(2,3,15,10) if compressed else f(2,3,15,10)
        for kind in [0x400a,0x400e,0x4010]:
            capture(f'fill shape {kind:x} {compressed}',[plus(kind,u(0x8060a020)+(u(1) if kind==0x400a else f(10,220) if kind==0x4010 else b'')+rect,0x8000|compressed)])
    for brushkind,payload in [(0,brush()),(1,u(V,1,3,0xff102030,0xff8090a0)),(4,u(V,4,0,0)+f(0,0,20,15)+u(0xff102030)+bytes(12)),(3,u(V,3,0,0,0xff102030)+f(4,5)+u(0)),(2,u(V,2))]:
        capture(f'brush {brushkind}',[define(1,payload),plus(0x400a,u(0,1)+f(2,3,15,10)),fill()])
    for style in range(7):
        pen=u(V,0,32,2)+f(1.5)+u(style)+brush()
        for kind in [0x400b,0x400d,0x400f,0x4011,0x4012,0x4019]:
            data=u(1)+f(2,3,15,10) if kind==0x400b else u(4)+f(2,2,10,3,12,14,20,5) if kind in [0x400d,0x4019] else (f(10,-220) if kind in [0x4011,0x4012] else b'')+f(2,3,15,10)
            capture(f'pen {style} shape {kind:x}',[define(2,pen),plus(kind,data)])
    for flags in [0,0x4000,0x800]:
        pts=h(2,2,18,3,10,16) if flags==0x4000 else bytes([2,2,16,1,0x78,13]) if flags==0x800 else f(2,2,18,3,10,16)
        types=bytes([0x41,0,0x41,1,0x41,0x81]) if flags==0x800 else bytes([0,1,0x81])
        path=u(V,3,flags)+pts+types
        capture(f'path {flags}',[define(3,path,1),plus(0x4014,u(0xff30a060),0x8001)])
    for kind in [0x402a,0x402c,0x402d,0x402e,0x402f]:
        payload=f(1,.2,-.1,1,2,3) if kind in [0x402a,0x402c] else f(2,3) if kind in [0x402d,0x402e] else f(25)
        for flags in [0,0x2000]:capture(f'transform {kind:x} {flags}',[plus(kind,payload,flags),fill()])
    for unit in range(8):capture(f'page unit {unit}',[plus(0x4030,f(.6),unit),fill()])
    for mode in range(7):capture(f'clip {mode}',[plus(0x4032,f(2,2,16,14)),plus(0x4032,f(6,5,14,13),mode<<8),fill()])
    capture('clip offset reset',[plus(0x4032,f(2,2,16,14)),plus(0x4035,f(2,1)),fill(),plus(0x4031),fill()])
    capture('save restore',[plus(0x4025,u(3)),plus(0x402d,f(2,3)),fill(),plus(0x4026,u(3)),fill()])
    capture('container',[plus(0x4027,f(2,3,18,14,0,0,30,20)+u(5),2),fill(),plus(0x4029,u(5)),fill()])
    capture('container no params',[plus(0x4028,u(8)),plus(0x402d,f(2,3)),fill(),plus(0x4029,u(8)),fill()])
    for index in [0,1,3,8]:capture(f'invalid restore {index}',[plus(0x4025,u(3)),plus(0x4026,u(index)),fill()])
    for kind in range(0x401d,0x4025):
        for flags in [0,1,4,9]:capture(f'rendering {kind:x} {flags}',[plus(kind,i(1,2) if kind==0x401d else b'',flags),fill()])
    for kind in [0x4013,0x4016,0x4017,0x4018,0x4037,0x4038,0x4039,0x4040]:capture(f'unsupported record {kind:x}',[fill(),plus(kind),fill()])
    # All variable pen fields must be consumed in spec order.
    pen=u(V,0,0x1fff,3)+f(2)+f(1,0,0,1,1,2)+u(1,2,1)+f(3)+u(3,1)+f(.5)+u(2)+f(2,1)+u(1,2)+f(0,1)+u(3)+b'abc'+u(2)+b'de'+brush()
    capture('pen all options',[define(2,pen),plus(0x400d,u(3)+f(2,3,12,16,20,5))])
    for flags,width,unit in [(0x2000,1,2),(0,-1,2),(0,1,7)]:
        capture(f'invalid pen {flags} {width} {unit}',[define(2,u(V,0,flags,unit)+f(width)+brush()),fill()])
    path=u(V,4,0)+f(2,2,4,15,15,15,20,3)+bytes([0,3,3,0x83])
    capture('path bezier clip',[define(3,path,1),plus(0x4033,b'',1),fill()])
    capture('unsupported clip region',[fill(),define(4,u(V),1),plus(0x4034,b'',1),fill()])
    for count in [0,1,2,1000001]:capture(f'polygon count {count}',[plus(0x400c,u(0xff102030,count)+f(1,2,3,4),0x8000)])
    for value in [float('nan'),float('inf'),-1,0]:capture(f'bad scale {value}',[plus(0x4030,f(value),2),fill()])
    for payload in [u(0,0,96,96),u(V,0,0,96),u(V,0,100001,96)]:capture('bad header '+payload.hex(),[plus(0x4001,payload),fill(),plus(0x4002)],False)
    payload=brush()
    for cut in range(1,len(payload)):
        capture(f'continued brush {cut}',[plus(0x4008,u(len(payload))+payload[:cut],0x8100),plus(0x4008,payload[cut:],0x100),plus(0x400a,u(0,1)+f(2,3,15,10))])
    capture('continued interrupted',[plus(0x4008,u(12)+b'xx',0x8100),fill()])
    capture('continued wrong slot',[plus(0x4008,u(12)+b'xx',0x8100),plus(0x4008,b'xx',0x101)])
    capture('continued short',[plus(0x4008,u(12)+b'xx',0x8100),plus(0x4008,b'xx',0x100)])
    for slot in [63,64,255]:capture(f'object slot {slot}',[define(1,brush(),slot),fill()])
    for tail in [[],[plus(0x4002)],[plus(0x4001,u(V,0,96,96)),fill()],[plus(0x4001,u(V,0,96,96)),fill(),plus(0x4002),fill()],[plus(0x4001,u(V,0,96,96)),plus(0x4001,u(V,0,96,96)),fill(),plus(0x4002)]]:capture(f'boundaries {len(cases)}',tail,False)
    for depth in [63,64,65]:capture(f'stack depth {depth}',[plus(0x4025,u(n)) for n in range(depth)]+[fill()])
    for align in range(3):
        for vertical in range(3):
            capture(f'text {align} {vertical}',[define(6,font(),0),define(7,fmt(0,align,vertical),1),plus(0x401c,u(0xff102030,1,5)+f(1,1,22,18)+'A\nB\rC'.encode('utf-16le'),0x8000)])
    for style in [0,1,2,4,8,15]:capture(f'text style {style}',[define(6,font(style=style)),plus(0x401c,u(0xff102030,0xffffffff,3)+f(1,1,22,18)+'AbC'.encode('utf-16le'),0x8000)])
    for flags,text in [(1,'abc'),(2,'abc'),(0,'a\tb'),(0x1000,'abcdefghij'),(0x4000,'abc')]:capture(f'text flags {flags} {text}',[define(6,font()),define(7,fmt(flags),1),plus(0x401c,u(0xff102030,1,len(text))+f(1,1,8,18)+text.encode('utf-16le'),0x8000)])
    for options in [0,1,2,3,5]:capture(f'driver {options}',[define(6,font()),plus(0x4036,u(0xff102030,options,0,3)+'ABC'.encode('utf-16le')+(f(2,10) if options&4 else f(2,10,9,11,16,12)),0x8000)])
    raw=bytes([20,30,40,80, 60,70,80,128, 100,110,120,255]*2)
    for pixel in [0x21808,0x22009,0x26200a,0xe200b,0x30101]:
        for stride in [12,-12]:capture(f'raw image {pixel:x} {stride}',[define(5,image(raw,0,3,2,stride,pixel)),draw_image()])
    rgba=Image.frombytes('RGBA',(3,2),bytes([20,30,40,80, 60,70,80,128, 100,110,120,255]*2))
    capture('image attributes',[define(5,image(raw,0,3,2,12,0x26200a)),define(8,u(V,0,3)+bytes(12),1),plus(0x401a,u(1,2)+f(0,0,3,2,2,2,18,14),0x2000)])
    capture('image clipped',[define(5,image(raw,0,3,2,12,0x26200a)),plus(0x4032,f(3,3,12,9)),draw_image(True)])
    capture('image redefine',[define(5,image(raw,0,3,2,12,0x26200a)),draw_image(),define(5,u(V,2)),draw_image(),fill()])
    for mode in ['RGBA','RGB','L','LA','P','1','I;16']:
        bitmap=rgba.convert(mode) if mode!='I;16' else Image.frombytes('I;16',(3,2),struct.pack('<6H',0,1,255,256,65535,1234))
        out=BytesIO();bitmap.save(out,format='PNG');capture(f'PNG {mode}',[define(5,image(out.getvalue())),draw_image(),draw_image(True)])
    for mode in ['L','P','I;16']:
        bitmap=rgba.convert(mode) if mode!='I;16' else Image.frombytes('I;16',(3,2),struct.pack('<6H',0,1,255,256,65535,1234))
        out=BytesIO();bitmap.save(out,format='PNG',transparency=1)
        capture(f'PNG transparency {mode}',[define(5,image(out.getvalue())),draw_image()])
    for mode in ['RGB','L','CMYK']:
        out=BytesIO();rgba.convert(mode).save(out,format='JPEG',quality=90);capture(f'JPEG {mode}',[define(5,image(out.getvalue())),draw_image()])
    import random
    rng=random.Random(731)
    bitmap=Image.frombytes('RGB',(19,17),rng.randbytes(19*17*3))
    for quality in [30,75,95]:
        for subsampling in [0,1,2]:
            for progressive in [False,True]:
                out=BytesIO();bitmap.save(out,format='JPEG',quality=quality,subsampling=subsampling,progressive=progressive)
                capture(f'JPEG noise {quality} {subsampling} {progressive}',[define(5,image(out.getvalue())),draw_image()])
    for progressive in [False,True]:
        out=BytesIO();bitmap.convert('CMYK').save(out,format='JPEG',quality=90,progressive=progressive)
        capture(f'JPEG CMYK noise {progressive}',[define(5,image(out.getvalue())),draw_image()])
    for format in ['GIF','BMP','TIFF','WEBP']:
        out=BytesIO();rgba.convert('RGB').save(out,format=format);capture(f'skipped image {format}',[define(5,image(out.getvalue())),draw_image(),fill()])
    for payload in [b'',b'bad png',b'\x89PNG\r\n\x1a\n',b'\xff\xd8']:
        capture(f'bad image {payload.hex()}',[define(5,image(payload)),draw_image()])
    # GetDC activates GDI only until the next EMF+ record; its state is temporary.
    capture_file('GetDC bridge',emf([comment([plus(0x4001,u(V,0,96,96)),plus(0x402d,f(2,3)),plus(0x4004)]),erec(43,i(1,1,15,12)),erec(35,f(2,0,0,2,0,0)),comment([fill(),plus(0x4002)])]))
    target=ROOT/'tests/fixtures/office_rust_reference_068/emfplus.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('EMF+ references:',len(cases),'files;',sum('document' in c for c in cases),'successful replay;',sum(len(c.get('outputs',{})) for c in cases),'output queries')
if __name__=='__main__': main()
