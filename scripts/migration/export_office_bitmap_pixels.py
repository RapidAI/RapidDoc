"""Original Pillow/metafile-render pixel references for native bitmap placement."""
import dataclasses,gzip,json,random
from pathlib import Path
from PIL import Image
from metafile_render.backends.bitmap import _affine_image_layer,_render_image_command
from metafile_render.backends.composite import _composite_image,_composite_path
from metafile_render.commands import DibPayload,DrawImageCommand
from metafile_render.primitives import Matrix,Rect
from metafile_render.geometry import FlattenBudget
from export_office_bitmap_reference import dib,bitmap_cases

def main():
    rng=random.Random(301030)
    source=Image.frombytes('RGBA',(5,3),bytes(rng.randrange(256) for _ in range(5*3*4)))
    affines=[]
    destinations=[((0,0),(10,0),(10,6),(0,6)),((.3,.7),(8.3,.7),(8.3,8.7),(.3,8.7)),((2,1),(9,4),(7,10),(0,7)),((10,8),(0,8),(0,0),(10,0)),((1,1),(2,2),(3,3),(2,2)),((100000,100000),(100001,100002),(100003,100005),(100002,100003)),((0,0),(.00001,.00002),(.00004,.00004),(.00003,.00002))]
    destinations += [tuple((rng.uniform(-7,14),rng.uniform(-7,14)) for _ in range(4)) for _ in range(100)]
    for destination in destinations:
        for matrix in [Matrix(),Matrix(1,.2,-.3,.7,1,-1)]:
            for mode in [3,4]:
                pixels=_affine_image_layer(source,destination,matrix,(11,9),mode)
                affines.append(dict(destination=destination,matrix=dataclasses.asdict(matrix),mode=mode,pixels=list(pixels.tobytes())))
    # Render the complete original image layer through decode -> source crop ->
    # alpha -> affine, with no clip stack; final path clipping is a later stage.
    layers=[]
    raw=bytes(rng.randrange(256) for _ in range(4*3*4))
    for alpha in [0,1,127,254,255]:
        for premultiplied in [False,True]:
            for mode in [3,4]:
                for coords in [None,(-1,-1,3,2),(3,2,0,0),(4,0,5,1),(.4,.2,3.2,2.8)]:
                    payload=DibPayload(dib(4,-3),raw,premultiplied)
                    command=DrawImageCommand(payload,((1,2),(10,3),(9,8),(0,7)),Rect(*coords) if coords else None,0xcc0020,mode,alpha)
                    output=_render_image_command(command,Matrix(),(11,9),FlattenBudget())
                    layers.append(dict(header=list(payload.header),bits=list(raw),use_source_alpha=premultiplied,source=dataclasses.asdict(command.source) if command.source else None,mode=mode,alpha=alpha,pixels=list(output.tobytes())))
    canvas=bytes(v for a in range(256) for b in range(256) for v in [(a*13+b)%256,(a+b*17)%256,(a*31+b*7)%256,a])
    layer=bytes(v for a in range(256) for b in range(256) for v in [(a*11+b*19)%256,(a*5+b*23)%256,(a+b*29)%256,b])
    composite=Image.frombytes('RGBA',(256,256),canvas)
    composite.alpha_composite(Image.frombytes('RGBA',(256,256),layer))
    # ROP dispatch uses a smaller subset; the source-over rounding separately
    # covers every pair of source/destination alpha values above.
    dispatch=[]
    for path,rops in [(False,[0,0xcc0020,0x8800c6,0xee0086,0x660046,0x550009,0x42,0xff0062,0x12345678]),(True,list(range(18)))]:
        for rop in rops:
            image=Image.frombytes('RGBA',(8,8),canvas[:256]);overlay=Image.frombytes('RGBA',(8,8),layer[:256])
            (_composite_path if path else _composite_image)(image,overlay,rop)
            dispatch.append(dict(path=path,rop=rop,pixels=list(image.tobytes())))
    root=Path(__file__).resolve().parents[2]
    from metafile_render.parser import prepare_metafile
    from metafile_render.records import iter_records
    from metafile_render.gdi.playback import _Playback
    from metafile_render.gdi.emf import _handle_emf_record
    from metafile_render.gdi.wmf import _handle_wmf_record
    from metafile_render.backends.raster import render_pillow
    from metafile_render.backends.session import RenderSession
    documents=[]
    def document_capture(data,name):
        try:
            header=prepare_metafile(data,dpi=200,size_hint=(16,12));p=_Playback(header)
            for record in iter_records(data,header.record_start,header.record_end,header.source_format):
                p.set_record_context(record.kind,record.index,record.offset)
                if header.source_format=='emf':_handle_emf_record(record.kind,record.data,p)
                else:_handle_wmf_record(record.kind,record.data.subreader(6,len(record.data)-6),p)
            document=p.finalize((16,12))
            if not document.commands:return
            pixels=render_pillow(document)
            documents.append(dict(name=name,hex=data.hex(),pixels=list(pixels.tobytes())))
        except Exception:return # Invalid inputs are already covered by replay references.
    bitmap_cases(document_capture)
    # Match eviction/hit statistics with the source cache at a small test limit.
    payloads=[DibPayload(dib(2,2),bytes([v]*16),False) for v in [20,40,60]]
    cache=[]
    with RenderSession(cache_limit=32) as session:
        from metafile_render.backends.bitmap import _decode_dib
        for index in [0,1,0,2,0,1,2]:
            payload=payloads[index]
            command=DrawImageCommand(payload,((0,0),(2,0),(2,2),(0,2)),None,0xcc0020)
            image=_decode_dib(command,session)
            cache.append(dict(index=index,hits=session.cache.hits,misses=session.cache.misses,used=session.cache.used,pixels=list(image.tobytes())))
    result=dict(source=list(source.tobytes()),affines=affines,layers=layers,canvas=list(canvas),layer=list(layer),source_over=list(composite.tobytes()),dispatch=dispatch,documents=documents,cache=cache)
    target=root/'tests/fixtures/office_rust_reference_068/metafile_bitmap_pixels.json.gz'
    target.write_bytes(gzip.compress(json.dumps(result,allow_nan=False).encode(),mtime=0))
    print('Pixel references:',len(affines),'affines;',len(layers),'layers; 65536 alpha pairs;',len(dispatch),'ROP dispatches;',len(documents),'bitmap documents;',len(cache),'cache queries')

if __name__=='__main__':main()
