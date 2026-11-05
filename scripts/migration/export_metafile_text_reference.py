"""Original metafile-render text/font references, never used by Rust runtime."""
import dataclasses,gzip,json,sys,itertools,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from export_office_path_pixels import obj,normalized_svg
def text_cases(capture):
    import struct
    from export_office_metafile_reference import wmf,wrec,emf,erec,h,i,u
    def efont(name='Arial',rotation=0,charset=1):
        r=bytearray(104);struct.pack_into('<II',r,0,82,104);struct.pack_into('<I5i',r,8,1,-17,0,rotation,rotation,400);r[33:36]=bytes([1,1,charset]);raw=name.encode('utf-16le');r[40:40+min(len(raw),62)]=raw[:62];return bytes(r)
    def etext(raw,options=0,advances=None,wide=True):
        count=len(raw)//2 if wide else len(raw);r=bytearray(76);struct.pack_into('<4i',r,8,0,0,90,60);struct.pack_into('<2i',r,36,12,30);struct.pack_into('<3I',r,44,count,76,options);struct.pack_into('<4i',r,56,2,2,85,55);r.extend(raw);r.extend(bytes(-len(r)%4));
        if advances is not None:struct.pack_into('<I',r,72,len(r));r.extend(struct.pack('<'+'i'*len(advances),*advances))
        struct.pack_into('<II',r,0,84 if wide else 83,len(r));return bytes(r)
    for alignment in [0,2,6,8,24,26,30,1,3,7,25,27,31]:
        for options,advances in [(0,None),(2,None),(4,[10,11,12]),(6,[10,11,12]),(8192,[10,1,11,-1,12,2]),(16,[10,11,12])]:
            capture(emf([efont(),erec(37,u(1)),erec(22,u(alignment)),erec(24,u(0x7814c8)),erec(25,u(0x0ab428)),etext('AVé'.encode('utf-16le'),options,advances)]),f'EMF text align {alignment} options {options}')
    for charset,raw in [(1,b'AV\x81\xe9'),(128,'日本語'.encode('shift_jis')),(134,'中文'.encode('gb18030')),(136,'漢語'.encode('big5')),(255,bytes([0x80,0x81,0xe0]))]:
        for rotation in [0,300,-900]:
            capture(emf([efont('Arial',rotation,charset),erec(37,u(1)),etext(raw,wide=False)]),f'EMF ANSI charset {charset} rotation {rotation}')
            font=h(-17,0,rotation,rotation,400)+bytes([0,1,1,charset])+bytes(4)+b'Arial\0'
            capture(wmf([wrec(0x2fb,font),wrec(0x12d,h(0)),wrec(0x521,struct.pack('<H',len(raw))+raw+bytes(len(raw)%2)+h(30,12))]),f'WMF TextOut charset {charset} rotation {rotation}')
    for alignment,options,dx in itertools.product([0,6,30,1],[0,2,4,6],[False,True]):
        raw=b'AVe';payload=h(30,12,3,options)+(h(2,2,85,55) if options&6 else b'')+raw+b'\0'+(h(10,11,12) if dx else b'')
        capture(wmf([wrec(0x12e,struct.pack('<H',alignment)),wrec(0xa32,payload)]),f'WMF ExtTextOut align {alignment} options {options} dx {dx}')
    for raw in [b'',b'\x00\xd8',b'A\0\x00\xd8', '𠀀'.encode('utf-16le')]:capture(emf([etext(raw)]),'EMF UTF16 '+raw.hex())
    for name in ['not-a-real-font','Calibri']:
        capture(emf([efont(name),erec(37,u(1)),etext('A'.encode('utf-16le')),etext('A'.encode('utf-16le'))]),'EMF font substitution '+name)
    for kind,payload in [(83,b''),(84,bytes(60)),(0x521,b'\x04\0A'),(0xa32,bytes(8))]:capture(emf([erec(kind,payload)]) if kind<100 else wmf([wrec(kind,payload)]),'truncated text '+str(kind))
def main():
    from metafile_render.primitives import Font,Color,Matrix,Rect,ClipOperation
    from metafile_render.fonts.resolve import resolve_font,_style_candidates
    from metafile_render.fonts.index import _font_aliases
    from metafile_render.geometry import FlattenBudget,rectangle_path
    from metafile_render.commands import DrawTextCommand,TextAlignment
    from metafile_render.backends.text import _render_text_command
    from metafile_render.backends.svg import _svg_text_elements
    from metafile_render.backends.session import RenderSession
    from metafile_render.gdi.objects import _decode_ansi_text
    from PIL import features
    fonts=[]
    for face,weight,italic,size in itertools.product(['Arial','Times New Roman','SimSun','not-a-real-font'],[400,700],[False,True],[9,17,32]):
        resolved=resolve_font(face,size,weight,italic,1);font=resolved.font
        for text in ['AV fi x+y','中文Δé','']:
            for anchor,start in [('la',(0.,0.)),('ms',(.25,.75)),('rd',(-.25,-.75))]:
                mask,offset=font.getmask2(text,anchor=anchor,start=start)
                fonts.append(dict(face=face,weight=weight,italic=italic,size=size,text=text,anchor=anchor,start=start,family=resolved.family,style=resolved.style,reason=resolved.reason,length=font.getlength(text),bounds=font.getbbox(text,anchor=anchor),mask_size=mask.size,offset=offset,mask=list(bytes(mask))))
    layers=[]
    for align,angle,opaque,positioned in itertools.product(['left','center','right'],[0.,30.,90.],[False,True],[False,True]):
        font=Font(face_name='Arial',height=-17,underline=True,strikeout=True)
        text='AVé';positions=((12.25,28.75),(24.25,28.75),(37.25,28.75)) if positioned else ()
        c=DrawTextCommand(text,(12.25,28.75),positions,font,17.,angle,TextAlignment(align,'baseline'),Color(120,20,200,173),Color(10,180,40,101),opaque,None,(ClipOperation(rectangle_path(Rect(0,0,70,50)),'and'),),(49.25,28.75) if positioned else None)
        budget=FlattenBudget()
        with RenderSession() as session:
            image=_render_text_command(c,Matrix(),(80,64),budget,session);svg=_svg_text_elements(c,Matrix(),session)
        request=obj(c);request['location']={'record_type':None,'record_index':None,'offset':None}
        layers.append(dict(command=request,pixels=list(image.tobytes()),svg=svg,used=budget.used))
    for height,vertical,angle,text in itertools.product([9.,32.,70.],['top','bottom','baseline'],[-45.,180.],['éx','A\nVé\n']):
        font=Font(face_name='Arial',height=-height,underline=True,strikeout=True)
        c=DrawTextCommand(text,(31.25,30.75),(),font,height,angle,TextAlignment('center',vertical),Color(120,20,200,173),Color(10,180,40,101),True,None,())
        budget=FlattenBudget()
        with RenderSession() as session:image=_render_text_command(c,Matrix(),(80,64),budget,session);svg=_svg_text_elements(c,Matrix(),session)
        request=obj(c);request['location']={'record_type':None,'record_index':None,'offset':None}
        layers.append(dict(command=request,pixels=list(image.tobytes()),svg=svg,used=budget.used))
    aliases=[]
    for face in ['Arial','Calibri','Times New Roman','SimSun','Microsoft YaHei','arial regular','Unicodeſ']:
        for charset in [1,128,134,255]:
            aliases.append(dict(face=face,charset=charset,aliases=_font_aliases(face,charset),styles=[dict(weight=w,italic=i,candidates=_style_candidates(face,w,i)) for w,i in [(400,False),(700,False),(400,True),(700,True)]]))
    codecs=[]
    for charset in [0,1,128,129,134,136,161,162,163,177,178,186,204,222,238,255]:
        for raw in [bytes(range(256)),b'AV\0Text\x81\x98\xff',bytes.fromhex('d6d0cec4a1a4a2e3'),bytes.fromhex('81308130fd39fd39')]:
            codecs.append(dict(charset=charset,raw=list(raw),text=_decode_ansi_text(raw,charset)))
    # Paired sequential decode checks validate replacement consumption, not just
    # table entries. All byte pairs plus deterministic byte strings and GB4 ranges.
    import random,struct
    rng=random.Random(301032);codec_checks=[]
    for charset in [128,129,134,136]:
        digest=hashlib.sha256()
        for n in range(65536):
            encoded=_decode_ansi_text(n.to_bytes(2,'big'),charset).encode('utf-8');digest.update(struct.pack('<I',len(encoded)));digest.update(encoded)
        codec_checks.append(dict(charset=charset,pairs_sha256=digest.hexdigest()))
        for _ in range(40):
            raw=rng.randbytes(rng.randrange(1,40));codecs.append(dict(charset=charset,raw=list(raw),text=_decode_ansi_text(raw,charset)))
    fixture=dict(freetype=features.version('freetype2'),raqm=features.check('raqm'),fonts=fonts,layers=layers,aliases=aliases,codecs=codecs,codec_checks=codec_checks)
    path=ROOT/'tests/fixtures/office_rust_reference_068/metafile_text.json.gz';path.write_bytes(gzip.compress(json.dumps(fixture,ensure_ascii=True).encode(),mtime=0))
    print('Text reference:',len(fonts),'font queries;',len(layers),'layers;',len(aliases),'font rules;',len(codecs),'codec inputs;',fixture['freetype'])
if __name__=='__main__':main()
