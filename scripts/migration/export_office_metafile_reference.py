"""Freeze installed metafile-render 0.3.0 file/geometry/GDI behavior.

Only references are Python. The Rust runtime does not consume serialized drawing
requests or call the original renderer. No new simplified drawing algorithm is
used as the test oracle.
"""
import dataclasses,gzip,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def wrec(kind,payload=b''):
    payload+=b'\0'*(len(payload)%2)
    return struct.pack('<IH',(6+len(payload))//2,kind)+payload
def wmf(records,placeable=True,objects=4):
    body=b''.join(records)+wrec(0)
    data=struct.pack('<HHHIHIH',1,9,0x300,(18+len(body))//2,objects,0,0)+body
    if placeable:
        head=struct.pack('<IHhhhhHI',0x9ac6cdd7,0,0,0,100,100,1440,0)
        checksum=0
        for i in range(0,20,2):checksum^=struct.unpack_from('<H',head,i)[0]
        data=head+struct.pack('<H',checksum)+data
    return data
def erec(kind,payload=b''):
    payload+=b'\0'*(-len(payload)%4)
    return struct.pack('<II',kind,8+len(payload))+payload
def emf(records,frame=(0,0,2540,1270),bounds=(0,0,100,50)):
    body=b''.join(records)+erec(14,b'\0'*12)
    head=bytearray(88);struct.pack_into('<II',head,0,1,88)
    struct.pack_into('<4i',head,8,*bounds);struct.pack_into('<4i',head,24,*frame)
    struct.pack_into('<IIII',head,40,0x464d4520,0x10000,88+len(body),len(records)+2)
    struct.pack_into('<4i',head,72,1920,1080,508,286)
    return bytes(head)+body
def plus(kind,payload=b'',flags=0):
    size=12+len(payload);padded=size+(-size%4)
    return struct.pack('<HHII',kind,flags,padded,len(payload))+payload+b'\0'*(padded-size)
def comment(records):
    data=struct.pack('<I',0x2b464d45)+b''.join(records)
    return erec(70,struct.pack('<I',len(data))+data)
def h(*values):return struct.pack('<'+'h'*len(values),*values)
def i(*values):return struct.pack('<'+'i'*len(values),*values)
def u(*values):return struct.pack('<'+'I'*len(values),*values)

def main():
    from metafile_render.parser import prepare_metafile
    from metafile_render.records import iter_records
    from metafile_render.gdi.playback import _Playback
    from metafile_render.gdi.emf import _handle_emf_record
    from metafile_render.gdi.wmf import _handle_wmf_record
    from metafile_render.primitives import Rect,Matrix
    from metafile_render import geometry as g
    from metafile_render.sizing import _bounded_canvas_size,check_render_size
    def obj(value):
        if dataclasses.is_dataclass(value):return obj(dataclasses.asdict(value))
        if isinstance(value,dict):return {k:obj(v) for k,v in value.items()}
        if isinstance(value,(bytes,bytearray,tuple,list)):return [obj(v) for v in value]
        return value
    cases=[]
    def capture(data,name,dpi=200,size=None):
        case=dict(name=name,hex=data.hex(),dpi=dpi,size=size)
        try:header=prepare_metafile(data,dpi=dpi,size_hint=size);case['header']=obj(header)
        except Exception as error:case['header_error']=getattr(error,'code','malformed');cases.append(case);return
        records=[]
        try:
            for record in iter_records(data,header.record_start,header.record_end,header.source_format):records.append(dict(kind=record.kind,index=record.index,offset=record.offset,size=len(record.data)))
            case['records']=records
        except Exception as error:case['records_error']=getattr(error,'code','malformed')
        if header.emfplus_mode=='only':
            from metafile_render.parser import parse_metafile
            case['emfplus']=True
            try:case['document']=obj(parse_metafile(data,dpi=dpi,size_hint=size))
            except Exception as error:case['replay_error']=getattr(error,'code','malformed')
            cases.append(case);return
        p=_Playback(header);trace=[];cursor=0
        try:
            for record in iter_records(data,header.record_start,header.record_end,header.source_format):
                p.set_record_context(record.kind,record.index,record.offset)
                if header.source_format=='emf':_handle_emf_record(record.kind,record.data,p)
                else:_handle_wmf_record(record.kind,record.data.subreader(6,len(record.data)-6),p)
                trace.append(dict(state=obj(p.state),path=obj(p.path_builder.build()),current=p.path_builder.current,start=p.path_builder.subpath_start,open=p.path_builder.figure_open,active=p.path_active,ready=p.path_ready,budget=obj(p.context.budget),new_commands=obj(p.commands[cursor:]),diagnostics=obj(p.diagnostics.freeze()),partial=p.partial))
                cursor=len(p.commands)
            case['document']=obj(p.finalize(size))
        except Exception as error:case['replay_error']=getattr(error,'code','malformed')
        case['trace']=trace;cases.append(case)
    rectangle=wrec(0x41b,h(50,80,10,5))
    capture(wmf([rectangle]),'WMF rectangle')
    capture(wmf([rectangle],False),'WMF standard bounds inferred')
    capture(wmf([],False),'WMF standard empty')
    for dpi in [1,96,144,200,1200]:capture(wmf([rectangle]),f'WMF DPI {dpi}',dpi)
    for kind in [0x418,0x41b,0x415,0x416]:capture(wmf([wrec(kind,h(50,80,10,5)),rectangle]),f'WMF rect/clip {kind}')
    capture(wmf([wrec(0x61c,h(20,30,50,80,10,5))]),'WMF roundrect')
    for kind in [0x817,0x81a,0x830]:capture(wmf([wrec(kind,h(50,20,20,80,60,90,5,10))]),f'WMF arc {kind}')
    for mode in range(1,10):capture(wmf([wrec(0x103,h(mode)),wrec(0x20b,h(10,20)),wrec(0x20c,h(50,100)),wrec(0x20d,h(5,7)),wrec(0x20e,h(200,300)),rectangle]),f'WMF map {mode}')
    capture(wmf([wrec(0x103,h(8)),wrec(0x20c,h(0,0)),rectangle]),'WMF zero window')
    capture(wmf([wrec(0x410,h(1,0,1,0)),wrec(0x412,h(2,3,-4,5)),wrec(0x20f,h(4,7)),wrec(0x211,h(-3,2)),rectangle]),'WMF scale and offsets')
    for level in [-10,-2,-1,0,1,2,5]:capture(wmf([wrec(0x127,h(level)),wrec(0x1e),wrec(0x1e),wrec(0x127,h(level)),rectangle]),f'WMF restore {level}')
    for style in [0,1,2,3,4,5,8,0x100,0x2000]:capture(wmf([wrec(0x2fa,struct.pack('<HhhI',style,3,0,0x443322)),wrec(0x12d,h(0)),rectangle]),f'WMF pen {style}')
    for index in range(21):capture(wmf([wrec(0x12d,struct.pack('<H',0x8000|index)),rectangle]),f'WMF stock {index}')
    capture(wmf([wrec(0x2fc,struct.pack('<HIH',2,0x123456,3)),wrec(0x12d,h(0)),wrec(0x1f0,h(0)),wrec(0x12d,h(0)),rectangle]),'WMF brush delete')
    for name in [b'Arial\0',b'\x81Arial\0',b' \x1cCourier\x1f ',b'\0ignored']:
        capture(wmf([wrec(0x2fb,h(-16,2,900,0,700)+bytes([1,1,1,134])+bytes(4)+name),wrec(0x12d,h(0)),rectangle]),'WMF LOGFONT '+name.hex())
    for kind in [0x324,0x325]:capture(wmf([wrec(kind,struct.pack('<H',3)+h(0,0,50,70,80,10))]),f'WMF polygon {kind}')
    capture(wmf([wrec(0x538,h(2,3,2)+h(0,0,10,20,30,0,40,40,50,30))]),'WMF polypolygon')
    capture(wmf([wrec(0x104,h(99)),wrec(0x626),wrec(0xffff),wrec(0x41f,struct.pack('<Ihh',0x112233,2,3)),rectangle]),'WMF partial diagnostics')
    capture(wmf([wrec(0x1e)]*65+[rectangle]),'WMF SaveDC resource limit')
    capture(wmf([wrec(0x415,h(50,80,10,5))]*65+[rectangle]),'WMF clip resource limit')
    capture(wmf([wrec(0x103,h(8)),wrec(0x20c,h(0,0))]+[rectangle]*70),'WMF diagnostics truncation')
    capture(wmf([wrec(0x324,h(100,1,2))]),'WMF bad point count')
    for kind in [42,43,44,45,46,47,55]:capture(emf([erec(kind,i(0,0,80,50)+(i(20,10) if kind==44 else i(70,20,10,30) if kind in [45,46,47,55] else b''))]),f'EMF shape {kind}')
    for mode in range(1,10):capture(emf([erec(17,i(mode)),erec(9,i(200,100)),erec(11,i(-400,300)),erec(10,i(10,20)),erec(12,i(5,7)),erec(43,i(0,0,80,50))]),f'EMF mapping {mode}')
    matrix=struct.pack('<6f',1,0.5,-0.25,2,10,-5)
    for mode in [1,2,3,4,7]:capture(emf([erec(35,matrix),erec(36,matrix+u(mode)),erec(43,i(0,0,80,50))]),f'EMF transform {mode}')
    capture(emf([erec(35,struct.pack('<6f',float('nan'),0,0,1,0,0))]),'EMF bad transform')
    for value in [0.,-1.,float('inf'),float('nan')]:capture(emf([erec(58,struct.pack('<f',value))]),f'EMF invalid miter {value}')
    for width in [10,30,50]:capture(emf([],frame=(0,0,width,width)),f'EMF ties-even frame {width}',dpi=127)
    capture(emf([],frame=(100,100,0,0)), 'EMF reversed physical frame')
    capture(emf([],frame=(0,0,0,0),bounds=(0,0,0,0)), 'EMF no visible bounds')
    capture(emf([]),'EMF size hint precedence',size=(13,17))
    for kind in [18,19,21]:capture(emf([erec(kind,u(0x10002)),erec(43,i(0,0,80,50))]),f'EMF full DWORD state {kind}')
    for kind in [2,3,4,5,6,85,86,87,88,89]:
        points=[(0,0),(10,40),(40,-10),(70,30)]
        raw=b''.join(struct.pack('<hh' if kind>=85 else '<ii',*p) for p in points)
        capture(emf([erec(kind,i(0,0,80,50)+u(len(points))+raw)]),f'EMF points {kind}')
    for kind in [7,8,90,91]:
        raw=b''.join(struct.pack('<hh' if kind>=85 else '<ii',*p) for p in [(0,0),(10,20),(30,0),(50,40),(70,0)])
        capture(emf([erec(kind,i(0,0,80,50)+u(2,5,3,2)+raw)]),f'EMF multipoly {kind}')
    capture(emf([erec(59),erec(27,i(10,10)),erec(54,i(40,50)),erec(61),erec(54,i(70,20)),erec(60),erec(63)]),'EMF path close/reopen')
    capture(emf([erec(60),erec(61),erec(62),erec(59),erec(59),erec(68)]),'EMF invalid path state')
    capture(emf([erec(59),erec(43,i(0,0,80,50)),erec(60),erec(67,u(5)),erec(26,i(2,-3)),erec(43,i(10,10,50,40)),erec(28)]),'EMF clipping path')
    region=u(32,1,2,32)+i(0,0,90,60)+i(0,0,40,40)+i(50,10,90,60)
    capture(emf([erec(75,u(len(region),1)+region),erec(43,i(0,0,80,50)),erec(75,u(0,5))]),'EMF region clipping')
    capture(emf([erec(41,i(10,20)+u(30)+struct.pack('<ff',30,140))]),'EMF angle arc')
    capture(emf([erec(41,i(10,20)+u(30)+struct.pack('<ff',100000,-400))]),'EMF large angle arc')
    capture(emf([erec(38,u(1,0x2001)+i(4,0)+u(0x123456)),erec(37,u(1)),erec(39,u(2,0x10001,0x123456,5)),erec(37,u(2)),erec(43,i(0,0,80,50)),erec(40,u(1)),erec(37,u(1))]),'EMF object handles/full brush style')
    capture(emf([erec(38,u(0,0)+i(1,0)+u(0))]),'EMF reserved handle')
    for count in [0,1,3,9]:
        capture(emf([erec(95,u(1,0,0,0,0,0x10007,5,0,0x123456,0,count)+u(1,3,0)),erec(37,u(1)),erec(43,i(0,0,80,50))]),f'EMF extended pen {count}')
    for dual in [0,1]:capture(emf([comment([plus(0x4001,u(0xdbc01001,0,96,96),dual),plus(0x4002)])]),f'EMF+ mode {dual}')
    capture(emf([comment([plus(0x4003)])]),'EMF+ missing header')
    capture(emf([comment([plus(0x4001,u(0xdbc01001,0,96,96))])]),'EMF+ missing EOF')
    capture(emf([comment([plus(0x4001,u(0xdbc01001,0,0,96)),plus(0x4002)])]),'EMF+ invalid DPI')
    for payload,label in [(wmf([rectangle]),'WMF'),(emf([]),'EMF')]:
        for size in [0,4,17,21,43,87,len(payload)-1]:capture(payload[:size],f'{label} truncation {size}')
    bad=bytearray(wmf([rectangle]));bad[20]^=1;capture(bytes(bad),'WMF checksum')
    capture(wmf([wrec(0),rectangle]),'WMF early EOF accepted as in source')
    capture(emf([erec(14,b'\0'*12),erec(43,i(0,0,80,50))]),'EMF early EOF accepted as in source')
    for count in [0,1,999,1_000_001]:
        bad=bytearray(emf([]));struct.pack_into('<I',bad,52,count);capture(bytes(bad),f'EMF declared record count {count}')
    for dpi in [0,1201]:capture(wmf([rectangle]),f'bad dpi {dpi}',dpi=dpi)
    from export_office_bitmap_reference import bitmap_cases
    bitmap_cases(capture)
    from export_metafile_text_reference import text_cases
    text_cases(capture)
    geometry=[]
    for coords in [(0,0,100,60),(100,60,0,0),(-5,-4,8,7),(0,0,0,1)]:
        rect=Rect(*coords)
        for kind in ['rectangle','ellipse','round_rectangle','arc']:
            for direction in ([1,2] if kind=='arc' else [2]):
                path={'rectangle':lambda:g.rectangle_path(rect),'ellipse':lambda:g.ellipse_path(rect),'round_rectangle':lambda:g.round_rectangle_path(rect,40,15),'arc':lambda:g.arc_path(rect,(100,0),(0,60),direction=direction,close_mode='pie')}[kind]()
                for flatness in [0.25,2.0]:
                    budget=g.FlattenBudget();flat=g.flatten_path(path,flatness=flatness,budget=budget)
                    matrix=Matrix(1,.25,-.5,2,7,-9)
                    geometry.append(dict(kind=kind,rect=coords,direction=direction,flatness=flatness,path=obj(path),flat=obj(flat),used=budget.used,bounds=obj(g.path_bounds(path)),closed=obj(g.close_open_subpaths(path)),transformed=obj(g.transform_path(path,matrix))))
    sizes=[]
    for requested in [(1,1),(8192,8192),(100000,100),(4000,4000),(4001,4001),(8000,2000),(0,10)]:
        case=dict(size=requested)
        try:case['bounded']=_bounded_canvas_size(requested)
        except Exception as error:case['error']=getattr(error,'code','malformed')
        sizes.append(case)
    target=ROOT/'tests/fixtures/office_rust_reference_068/metafile_geometry.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,geometry=geometry,sizes=sizes),ensure_ascii=True,allow_nan=False).encode(),mtime=0))
    print('Metafile references:',len(cases),'files;',sum(len(c.get('trace',[])) for c in cases),'record snapshots;',len(geometry),'geometry;',len(sizes),'canvas cases')
if __name__=='__main__':main()
