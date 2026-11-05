"""Freeze original metafile-render DIB records, validation, alpha and cropping.

Reference-only Python; the Rust runtime has no Python dependency.
"""
import dataclasses
import gzip
import json
import struct
from pathlib import Path
from export_office_metafile_reference import wmf, wrec, emf, erec

ROOT = Path(__file__).resolve().parents[2]

def dib(width=2, height=2, bpp=32, compression=0):
    result = bytearray(40)
    struct.pack_into('<IiiHHI', result, 0, 40, width, height, 1, bpp, compression)
    return bytes(result)

def bitmap_cases(capture):
    bits = bytes([10,20,30,255,40,50,60,128,70,80,90,0,100,110,120,255])
    header = dib()
    for rop in [0xcc0020,0x8800c6,0xee0086,0x660046,0x550009,0x42,0xff0062,0,0x12345678]:
        payload = struct.pack('<IH8h',rop,0,2,2,-1,0,-5,10,3,4)+header+bits
        capture(wmf([wrec(0xf43,payload)]),f'WMF StretchDIB ROP {rop:x}')
        capture(wmf([wrec(0xf43,payload)],False),f'WMF DIB inferred bounds {rop:x}')
    for kind in [76,77,80,81,114]:
        for variant in ['normal','empty','truncated','bad-offset','unknown-rop','flipped','overflow-sum','alpha']:
            record = bytearray(108)
            struct.pack_into('<II',record,0,kind,len(record)+len(header)+len(bits))
            struct.pack_into('<4i',record,24,4,3,10,-5)
            if kind in [80,81]:
                struct.pack_into('<4i',record,32,0,-1,2,2)
                struct.pack_into('<4I',record,48,108,len(header),148,len(bits))
                struct.pack_into('<Iii',record,68,0xcc0020,10,-5)
                offsets = (48,52,56,60)
            else:
                struct.pack_into('<I',record,40,0xcc0020)
                struct.pack_into('<2i',record,44,0,-1)
                struct.pack_into('<4I',record,84,108,len(header),148,len(bits))
                struct.pack_into('<2i',record,100,2,2)
                offsets = (84,88,92,96)
            if kind == 114:record[40:44] = bytes([0,0,128,1])
            if variant == 'empty':struct.pack_into('<I',record,offsets[1],0)
            if variant == 'truncated':struct.pack_into('<I',record,offsets[3],1)
            if variant == 'bad-offset':struct.pack_into('<I',record,offsets[0],0xffffffff)
            if variant == 'unknown-rop' and kind not in [80,114]:struct.pack_into('<I',record,68 if kind==81 else 40,0x12345678)
            if variant == 'flipped':struct.pack_into('<2i',record,24,-10,20)
            if variant == 'overflow-sum':struct.pack_into('<2i',record,24,2147483647,2147483647)
            capture(emf([bytes(record)+header+bits]),f'EMF bitmap {kind} {variant}')
    for payload in [b'',bytes(22),struct.pack('<IH8h',0xcc0020,0,*([0]*8))+dib()+b'X']:
        capture(wmf([wrec(0xf43,payload)]),'WMF invalid DIB '+str(len(payload)))

def main():
    from metafile_render.commands import DibPayload,DrawImageCommand
    from metafile_render.dib import validate_dib_payload
    from metafile_render.gdi.bitmap import _dib_bits_offset
    from metafile_render.backends.bitmap import _decode_raw_alpha_dib,_decode_image_payload,_crop_source,_crop_destination
    from metafile_render.primitives import Rect
    from PIL import Image
    cases=[]
    def capture(header,bits,name):
        payload=DibPayload(header,bits,use_source_alpha=True)
        case=dict(name=name,header=list(header),bits=list(bits))
        for key,fn in [('validate',lambda:validate_dib_payload(payload)),('offset',lambda:_dib_bits_offset(header)),('alpha',lambda:_decode_raw_alpha_dib(DrawImageCommand(payload,((0,0),(2,0),(2,2),(0,2)),None,0)))]:
            try:
                value=fn()
                case[key]=list(value.tobytes()) if isinstance(value,Image.Image) else value
            except Exception as error:case[key+'_error']=getattr(error,'code','malformed')
        cases.append(case)
    # Exhaust every alpha value, including invalid premultiplied channels.
    bits=bytes(v for a in range(256) for c in [0,1,63,127,128,254,255] for v in [c,(c+17)%256,(c+93)%256,a])
    for height in [1,-1]:capture(dib(len(bits)//4,height),bits,f'all alpha values height {height}')
    for height in [2,-2]:capture(dib(height=height),bytes(range(16)),f'row orientation {height}')
    for bpp in [1,4,8,16,24,32,0,3,64]:
        for compression in [0,1,2,3,4,6]:capture(dib(bpp=bpp,compression=compression),bytes(range(32)),f'bpp {bpp} compression {compression}')
    for width,height in [(0,2),(-1,2),(2,0),(2,-2147483648),(2147483647,2)]:capture(dib(width,height),b'',f'invalid size {width} {height}')
    for length in [0,3,11,12,16,39]:capture(dib()[:length],b'',f'truncated header {length}')
    core=struct.pack('<I4H',12,2,2,1,8)
    capture(core,bytes(8),'core indexed')
    for length in [0,1,15]:capture(dib(),bytes(length),f'truncated bits {length}')
    for compression in [3,6]:capture(dib(compression=compression)+bytes(16),bytes(16),f'external masks {compression}')
    decodes=[]
    def decode_capture(header,bits,name,alpha=False):
        case=dict(name=name,header=list(header),bits=list(bits),use_source_alpha=alpha)
        try:
            image=_decode_image_payload(DrawImageCommand(DibPayload(header,bits,alpha),((0,0),(2,0),(2,2),(0,2)),None,0))
            case.update(size=image.size,pixels=list(image.tobytes()))
        except Exception as error:case['error']=getattr(error,'code','malformed')
        decodes.append(case)
    for case in cases:
        for alpha in [False,True]:decode_capture(bytes(case['header']),bytes(case['bits']),case['name']+str(alpha),alpha)
    for bpp in [1,4,8,16,24,32]:
        for size in [12,40,52,56,64,108,124,44]:
            for height in [2,-2]:
                if size==12:
                    if height<0:continue
                    header=struct.pack('<I4H',12,5,height,1,bpp)
                else:
                    header=bytearray(dib(5,height,bpp));header.extend(bytes(size-40));struct.pack_into('<I',header,0,size);header=bytes(header)
                if bpp<=8:header+=bytes(v for i in range(1<<bpp) for v in ([i*31%256,i*17%256,i*47%256] + ([] if size==12 else [0])))
                bits=bytes((i*37+19)%256 for i in range(((5*bpp+31)//32)*4*2))
                decode_capture(header,bits,f'raw bpp {bpp} header {size} height {height}')
    masks32=[(0xff0000,0xff00,0xff,0),(0xff000000,0xff0000,0xff00,0),(0xff000000,0xff00,0xff,0),(0xff000000,0xff0000,0xff00,0xff),(0xff,0xff00,0xff0000,0xff000000),(0xff0000,0xff00,0xff,0xff000000),(0xff000000,0xff00,0xff,0xff0000),(0,0,0,0),(1,2,4,8)]
    for bpp,masks in [(16,(0xf800,0x7e0,0x1f,0)),(16,(0x7c00,0x3e0,0x1f,0)),(24,(0xff0000,0xff00,0xff,0)),(16,(0x7c00,0x3e0,0x1f,0x8000)),(24,(0xff0000,0xff00,0xff,0xff000000))]+[(32,m) for m in masks32]:
        for size in [40,52,56,64,108,124]:
            header=bytearray(dib(4,2,bpp,3));struct.pack_into('<I',header,0,size)
            header.extend(struct.pack('<4I',*masks)[:12 if size in [40,52] else 16]);header.extend(bytes(max(0,size-len(header))))
            for alpha in [False,True]:decode_capture(bytes(header),bytes(i*19%256 for i in range(((4*bpp+31)//32)*8)),f'masks {bpp} {masks} header {size} alpha {alpha}',alpha)
    for bpp in [1,4,8]:
        for used in [1,2,3,16,256,257]:
            for gray in [False,True]:
                header=bytearray(dib(4,2,bpp));struct.pack_into('<I',header,32,used)
                palette=bytes(v for i in range(used) for v in ([0 if i==0 else 255]*3+[0] if gray and used==2 else [i%256]*3+[0] if gray else [i*13%256,i*19%256,i*41%256,0]))
                for length in sorted(set([0,1,2,3,5,6,7,len(palette)//2,len(palette)])):
                    decode_capture(bytes(header)+palette[:length],bytes(range(32)),f'palette bpp {bpp} used {used} gray {gray} length {length}')
    for compression in [1,2]:
        for payload in [bytes([4,0x12,0,0,4,0x34,0,1]),bytes([0,4,0x12,0x34,0x56,0x78,0,0,4,0x21,0,1]),bytes([0,5,0x12,0x34,0x56,0,0,0,4,0x12,0,1]),bytes([9,0x12,0,0,9,0x34]),bytes([0,2,1,1,3,0x21,0,1]),bytes([0,1]),bytes([1,0x12]),bytes([0,2,255,255]),bytes([0,0,0,0])]:
            bpp=8 if compression==1 else 4
            palette=bytes(v for i in range(1<<bpp) for v in [i*13%256,i*19%256,i*41%256,0])
            header=dib(4,2,bpp,compression)+palette
            decode_capture(header,payload,f'RLE{bpp} {payload.hex()}')
            decode_capture(header+b'X',payload,f'RLE{bpp} odd file offset {payload.hex()}')
            for used in [1,2,3,16]:
                header=bytearray(dib(4,2,bpp,compression));struct.pack_into('<I',header,32,used)
                palette=bytes(v for i in range(used) for v in ([0 if i==0 else 255]*3+[0] if used==2 else [i]*3+[0]))
                decode_capture(bytes(header)+palette,payload,f'RLE{bpp} gray palette {used} {payload.hex()}')
    for bpp in [1,4]:
        for width in [5,9,17]:
            header=bytearray(dib(width,2,bpp));struct.pack_into('<I',header,32,3)
            palette=bytes(v for i in range(3) for v in [i,i,i,0])
            decode_capture(bytes(header)+palette,bytes(range(128)),f'gray raw stride bpp {bpp} width {width}')
    crops=[]
    image=Image.frombytes('RGBA',(4,3),bytes(v for y in range(3) for x in range(4) for v in [x,y,17,255]))
    destination=((10.,5.),(40.,15.),(35.,45.),(5.,35.))
    for coords in [None,(0,0,4,3),(-1,-2,3,2),(5,0,8,3),(3,2,-1,-2),(.3,.4,2.2,1.8),(1,1,1,2),(-10,-10,10,10),(0,0,1e-10,1)]:
        source=None if coords is None else Rect(*coords)
        cropped,fraction=_crop_source(image,source)
        crops.append(dict(source=dataclasses.asdict(source) if source else None,fraction=dataclasses.asdict(fraction),pixels=list(cropped.tobytes()) if cropped else None,destination=_crop_destination(destination,fraction)))
    from metafile_render.backends.composite import _paste_bitwise
    operations=['xor','and','or','not_or','and_not_source','not_source','source_and_not_destination','not_and','not_xor','destination_or_not_source','source_or_not_destination','invert','black','white']
    destination=bytes(v for a in range(256) for c in [0,1,63,127,128,254,255] for v in [c,(c+17)%256,(c+93)%256,a])
    layer=bytes(v for a in range(256) for c in [0,1,63,127,128,254,255] for v in [(c+33)%256,(c+65)%256,(c+139)%256,(a*137+c)%256])
    composites=[]
    for operation in operations:
        canvas=Image.frombytes('RGBA',(len(destination)//4,1),destination)
        _paste_bitwise(canvas,Image.frombytes('RGBA',canvas.size,layer),operation)
        composites.append(dict(operation=operation,pixels=list(canvas.tobytes())))
    target=ROOT/'tests/fixtures/office_rust_reference_068/metafile_bitmap.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,decodes=decodes,crops=crops,composites=composites,canvas=list(destination),layer=list(layer)),allow_nan=False).encode(),mtime=0))
    print('Bitmap references:',len(cases),'payloads;',len(decodes),'decodes;',len(crops),'crop cases;',len(composites),'logical operations')

if __name__=='__main__':main()
