"""Freeze installed metafile-render/Pillow path raster behavior, reference only."""
import dataclasses,gzip,json,random,itertools
from pathlib import Path
from PIL import Image,ImageDraw
from metafile_render.backends import paths as p
from metafile_render.backends.session import RenderSession
from metafile_render.primitives import Rect,Matrix,Pen,Brush,Color,ClipOperation
from metafile_render.commands import DrawPathCommand
from metafile_render.geometry import PathBuilder,FlattenBudget,rectangle_path,ellipse_path,round_rectangle_path

def obj(value):
    if dataclasses.is_dataclass(value):return obj(dataclasses.asdict(value))
    if isinstance(value,dict):return {k:obj(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,bytes)):return [obj(v) for v in value]
    return value
import re,base64,hashlib
from io import BytesIO
def normalized_svg(raw):
    images=[]
    def replace(match):
        image=base64.b64decode(match.group())
        with Image.open(BytesIO(image)) as img:
            summary=dict(size=list(img.size),pixels_sha256=hashlib.sha256(img.convert('RGBA').tobytes()).hexdigest(),ppm=[round(v/0.0254) for v in img.info['dpi']] if 'dpi' in img.info else None)
        images.append(summary);return f'PNG[{len(images)-1}]'
    text=raw.decode()
    text=re.sub(r'(?<=data:image/png;base64,)[A-Za-z0-9+/=]+',replace,text)
    text=re.sub(r'(?<=data-mime="image/png">)[A-Za-z0-9+/=]+',replace,text)
    return dict(markup=text,images=images)

def main():
    rng=random.Random(301031);size=(19,17)
    polygons=[]
    coords=[[(0,0),(5,0),(10,0),(10,10),(0,10)],[(2,2),(8,8),(14,2),(14,14),(2,14)],[(1.5,2.7),(12.9,2.2),(4.1,14.9)],[(1,1),(2,2),(3,3)],[(0,0),(0,0),(0,0)]]
    coords += [[(rng.uniform(-10,25),rng.uniform(-10,25)) for _ in range(rng.randrange(3,15))] for _ in range(300)]
    for points in coords:
        mask=Image.new('L',size);ImageDraw.Draw(mask).polygon(points,fill=255)
        polygons.append(dict(points=points,pixels=list(mask.tobytes())))
    subpaths=[]
    shapes=[([(1,1),(16,1),(16,14),(1,14),(1,1)],True), ([(3,3),(3,12),(12,12),(12,3),(3,3)],True), ([(5,5),(10,5),(10,10),(5,10),(5,5)],True)]
    inputs=[[shapes[0]],[shapes[0],shapes[1]],[*shapes],[shapes[0],(list(reversed(shapes[1][0])),True)], [([(0,0),(8,8),(0,8),(8,0)],True)], [([(1,1),(2,2),(3,3)],True)],[]]
    inputs += [[(points,False)] for points in coords[5:80]]
    for paths in inputs:
        for rule in ['evenodd','nonzero']:
            case=dict(paths=paths,rule=rule);budget=FlattenBudget()
            try:case['pixels']=list(p._subpath_mask(paths,size,rule,budget).tobytes())
            except Exception as e:case['error']=getattr(e,'code','malformed')
            case['used']=budget.used;subpaths.append(case)
    strokes=[];dashes=[]
    lines=[([(2,2),(15,3),(8,14)],False),shapes[0], ([(4,4),(4,4),(12,12)],False), ([(2,5),(15,5),(2,5)],False)]
    for points,closed in lines:
        for pattern in [(),(4.,2.),(.2,.3),(1.,2.,3.),(-1.,0.),(5.,)]:
            dashes.append(dict(points=points,closed=closed,pattern=pattern,visible=p._dash_polyline(points,dashes=pattern,closed=closed)))
    for (points,closed),cap,join,width in itertools.product(lines,['round','square','flat'],['round','bevel','miter'],[1.,2.5,8.]):
        pen=Pen(cap=cap,join=join)
        for pattern in [(),(4.,2.)]:
            budget=FlattenBudget();mask=p._stroke_mask([(points,closed)],size=size,width=width,pen=pen,dashes=pattern,miter_limit=2.,budget=budget)
            strokes.append(dict(points=points,closed=closed,pen=obj(pen),width=width,pattern=pattern,pixels=list(mask.tobytes()),used=budget.used))
    hatches=[]
    for size2 in [(1,1),(7,5),(19,17),(16,16)]:
        for hatch in range(8):
            brush=Brush(kind='hatch',hatch=hatch,color=Color(40,80,120,127))
            hatches.append(dict(size=size2,brush=obj(brush),pixels=list(p._hatch_layer(size2,brush).tobytes())))
    clips=[]
    a=rectangle_path(Rect(1,1,13,13));b=ellipse_path(Rect(5,4,18,16))
    for mode1,mode2 in itertools.product(['copy','and','or','xor','diff'],repeat=2):
        clip=(ClipOperation(a,mode1),ClipOperation(b,mode2,'nonzero'))
        budget=FlattenBudget()
        with RenderSession() as session:
            mask=p._clip_mask(clip,Matrix(),size,budget,session);used=budget.used
            cached=p._clip_mask(clip,Matrix(),size,budget,session)
            clips.append(dict(clip=obj(clip),pixels=list(mask.tobytes()),used=used,used_after_hit=budget.used,hits=session.cache.hits,misses=session.cache.misses,cache_used=session.cache.used))
    commands=[]
    clip=(ClipOperation(a,'copy'),ClipOperation(b,'xor'))
    paths=[rectangle_path(Rect(2,2,15,14)),ellipse_path(Rect(2,2,15,14)),round_rectangle_path(Rect(2,2,15,14),4,6)]
    for path,cosmetic,brush_kind,scale in itertools.product(paths,[True,False],['solid','hatch','null'],[1,2]):
        pen=Pen(color=Color(220,10,30,160),width=2.5,cosmetic=cosmetic,cap='square',join='miter',dashes=(3.,1.))
        brush=Brush(kind=brush_kind,color=Color(20,200,80,100),hatch=4)
        for clipping in [(),clip]:
            command=DrawPathCommand(path,pen,brush,True,True,'nonzero',clipping,13,2.)
            budget=FlattenBudget()
            matrix=Matrix(1.1,.2,-.1,.9,1,-1)
            with RenderSession() as session:
                layer=p._render_path_command(command,matrix,size,scale,budget,session)
            request=obj(command);request['location']={'record_type':None,'record_index':None,'offset':None}
            commands.append(dict(command=request,matrix=obj(matrix),scale=scale,pixels=list(layer.tobytes()),used=budget.used))
    target=Path(__file__).resolve().parents[2]/'tests/fixtures/office_rust_reference_068/metafile_paths.json.gz'
    from metafile_render.parser import prepare_metafile
    from metafile_render.records import iter_records
    from metafile_render.gdi.playback import _Playback
    from metafile_render.gdi.emf import _handle_emf_record
    from metafile_render.gdi.wmf import _handle_wmf_record
    from metafile_render.backends.raster import _render_pillow_once
    from export_office_bitmap_reference import bitmap_cases
    from export_office_metafile_reference import wmf,wrec,emf,erec,h,i
    documents=[]
    def capture(data,name):
        for scale in [1,2]:
            case=dict(name=name,hex=data.hex(),scale=scale)
            try:
                header=prepare_metafile(data,dpi=200,size_hint=(24,20));pback=_Playback(header)
                for record in iter_records(data,header.record_start,header.record_end,header.source_format):
                    pback.set_record_context(record.kind,record.index,record.offset)
                    if header.source_format=='emf':_handle_emf_record(record.kind,record.data,pback)
                    else:_handle_wmf_record(record.kind,record.data.subreader(6,len(record.data)-6),pback)
                doc=pback.finalize((24,20))
                with RenderSession() as session:
                    rendered=_render_pillow_once(doc,scale,session)
                    case.update(pixels=list(rendered.tobytes()),used=session.flatten.used,hits=session.cache.hits,misses=session.cache.misses,cache_used=session.cache.used)
            except Exception as e:case['error']=getattr(e,'code','malformed')
            documents.append(case)
    original=json.loads(gzip.decompress(target.with_name('metafile_geometry.json.gz').read_bytes()))
    for c in original['cases']:
        if 'document' in c and not (c.get('pending_emfplus') or c.get('emfplus')):capture(bytes.fromhex(c['hex']),c['name'])
    def mixed(data,name):
        header=prepare_metafile(data,dpi=200,size_hint=None)
        records=[r for r in iter_records(data,header.record_start,header.record_end,header.source_format) if r.kind not in ([0] if header.source_format=='wmf' else [1,14])]
        raw=[r.data.bytes(0,len(r.data)) for r in records]
        if header.source_format=='wmf':
            capture(wmf([wrec(0x41b,h(10,12,0,0)),wrec(0x1e),wrec(0x416,h(8,10,-1,-1)),*raw,wrec(0x418,h(12,16,2,2)),wrec(0x127,h(-1)),wrec(0x41b,h(14,18,6,6))]),'mixed '+name)
        else:
            capture(emf([erec(43,i(0,0,12,10)),erec(33),erec(30,i(-1,-1,10,8)),*raw,erec(42,i(2,2,16,12)),erec(34,i(-1)),erec(43,i(6,6,18,14))]),'mixed '+name)
    bitmap_cases(mixed)
    from metafile_render.backends.raster import render_pillow
    from metafile_render.sizing import choose_raster_scale
    from PIL import ImageFilter
    from metafile_render.backends.svg import render_svg,_svg_number
    import re,base64,hashlib
    from io import BytesIO
    filters=[]
    for source_size,target_size in [((1,1),(3,5)),((9,7),(3,2)),((4,8),(4,2)),((12,5),(3,5)),((96,80),(24,20)),((3,2),(11,9))]:
        pixels=bytes(rng.randrange(256) for _ in range(source_size[0]*source_size[1]*4));image=Image.frombytes('RGBA',source_size,pixels)
        resized=image.resize(target_size,Image.Resampling.LANCZOS)
        filters.append(dict(size=source_size,target=target_size,pixels=list(pixels),resize=list(resized.tobytes()),unsharp=list(resized.filter(ImageFilter.UnsharpMask(radius=.6,percent=100,threshold=2)).tobytes())))
    final=[]
    for c in documents[::2]:
        if 'error' in c:continue
        data=bytes.fromhex(c['hex']);header=prepare_metafile(data,dpi=200,size_hint=(24,20));pback=_Playback(header)
        for record in iter_records(data,header.record_start,header.record_end,header.source_format):
            pback.set_record_context(record.kind,record.index,record.offset)
            if header.source_format=='emf':_handle_emf_record(record.kind,record.data,pback)
            else:_handle_wmf_record(record.kind,record.data.subreader(6,len(record.data)-6),pback)
        doc=pback.finalize((24,20))
        with RenderSession() as session:
            pixels=render_pillow(doc,session)
            result=dict(name=c['name'],hex=c['hex'],pixels=list(pixels.tobytes()),scale=choose_raster_scale(doc,maximum=4),used=session.flatten.used)
        with RenderSession() as session:result['svg']=normalized_svg(render_svg(doc,session))
        final.append(result)
    # Tight flatten budgets also cross the native callback; no exception may
    # unwind through C++, and the original post-charge count must be retained.
    budgets=[]
    for limit in [0,1,4,10,20,40]:
        budget=FlattenBudget(limit=limit)
        case=dict(limit=limit)
        try:case['pixels']=list(p._path_mask(b,Matrix(),size,'evenodd',budget).tobytes())
        except Exception as e:case['error']=getattr(e,'code','malformed')
        case['used']=budget.used;budgets.append(case)
    numbers=[dict(value=v,output=_svg_number(v)) for v in [0.,-0.,1.,-1.,.00005,.00015,.00025,-.00005,1.23445,1.23455,999999.12345,1e12+.125]]
    target.write_bytes(gzip.compress(json.dumps(dict(size=size,polygons=polygons,subpaths=subpaths,dashes=dashes,strokes=strokes,hatches=hatches,clips=clips,commands=commands,documents=documents,budgets=budgets,budget_path=obj(b),filters=filters,final=final,numbers=numbers),allow_nan=False).encode(),mtime=0))
    print('Path references:',len(polygons),'polygons;',len(subpaths),'fills;',len(dashes),'dash cases;',len(strokes),'strokes;',len(hatches),'hatches;',len(clips),'clip stacks;',len(commands),'commands;',len(documents),'document renders;',len(budgets),'budget boundaries')
if __name__=='__main__':main()
