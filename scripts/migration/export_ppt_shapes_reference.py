"""Original PPT grouped geometry, table membership and bound shape text."""
import gzip,json,random,struct,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt import records as r,parser as p,style_text as s
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    from loguru import logger
    logger.remove();rng=random.Random(731526);cases=[];props=[];numbers=[]
    def rec(kind,payload=b'',version=0,instance=0):return struct.pack('<HHI',(instance<<4)|version,kind,len(payload))+payload
    def add(name,data,masters=None,external=None,links=None):
        masters=masters or {};external=external or [];links=links or {};c=dict(name=name,data=data.hex(),masters={k:[asdict(x)for x in v]for k,v in masters.items()},external=[asdict(x)for x in external],links=links)
        budget=r.RecordBudget();record=r.record_at(data,0,budget=budget)
        try:
            shapes=p._collect_shapes(record,budget)
            c['shapes']=[dict(key=x.key,offset=x.record.offset,space=asdict(x.space)if x.space else None,bbox=x.bbox,order=x.order,group_key=x.group_key)for x in shapes.shapes]
            c['groups']=[dict(key=g.key,offset=g.group_shape.offset,members=[x.key for x in g.shapes],order=g.order,authoritative=g.authoritative)for g in shapes.table_groups]
            c['texts']=[asdict(content)if(content:=p._shape_text_content(shape,external,masters,links,budget))else None for shape in shapes.shapes]
            tables=[]
            for group in shapes.table_groups:
                table=p._table_from_group(group,external,masters,links,budget)
                if table:
                    value=asdict(table);value['shape_offsets']=sorted(table.shape_offsets);tables.append(value)
                else:tables.append(None)
            c['tables']=tables
            c['record_visits']=budget.count
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    for i in range(150):
        leaves=[]
        for j in range(rng.randrange(0,8)):
            anchor=rec(0xf010,struct.pack('<4h',rng.randrange(-1000,1000),rng.randrange(-1000,1000),rng.randrange(1000,5000),rng.randrange(1000,5000))) if i%2 else rec(0xf00f,struct.pack('<4i',0,0,rng.choice([900,900000]),rng.choice([500,500000])))
            text='text'+str(j)+' 😀\rbody\vline'
            content=rec(0xf9f,struct.pack('<I',rng.choice([0,1,4,7])))+rec(0xfa0,text.encode('utf-16le'))
            if i%3==0:content=rec(0xf9e,struct.pack('<I',rng.randrange(0,5)))
            textbox=rec(0xf00d,content,15)
            leaf=rec(0xf004,rec(0xf00a,struct.pack('<II',j,0),2,1)+anchor+textbox,15)
            leaves.append(leaf)
        body=b''.join(leaves)
        for depth in range(i%4):
            coords=rec(0xf009,struct.pack('<4i',100,200,rng.choice([50,1000]),rng.choice([100,2000])))
            group=rec(0xf004,coords+rec(0xf00f,struct.pack('<4i',100,200,1800,2000))+rec(0xf00b,struct.pack('<HI',0x39f,i%2),3,1),15)
            body=rec(0xf003,group+body,15)
        masters={4:[s.MasterLevel(bullet=i%2==0,bold=i%3==0)]}
        external=[p._TextContent(tuple(p._build_paragraphs('external '+str(j),s.StyleRuns(),[],[])),4)for j in range(3)]
        add('group/'+str(i),rec(1006,body,15),masters,external)
    for enabled in [-1,0,1,2]:
        for slot in [0,1,7,15]:
            payload=b''.join(struct.pack('<IhHHII',0x3000000,enabled,3,5+i,0,0)for i in range(16))
            text='numbered item';count=len(text)+1
            style=struct.pack('<IHIH',count,0,1,1)+struct.pack('<IIH',count,1,slot<<10)
            textbox=rec(0xf00d,rec(0xfa8,text.encode())+rec(0xfa1,style),15)
            tag=rec(0x138a,rec(0xfba,'___PPT9'.encode('utf-16le'))+rec(0x138b,rec(0xfac,payload)),15)
            leaf=rec(0xf004,rec(0xf00f,struct.pack('<4i',0,0,500,300))+textbox+tag,15)
            add('numbered-shape/'+str(enabled)+'/'+str(slot),rec(1006,leaf,15))
    for mode in ['grid','merge-h','merge-v','hole','overlap','unmarked','line','wrong-shape','jitter','empty','three-cells']:
        leaves=[]
        for row in range(3):
            for col in range(3):
                if mode=='hole' and (row,col)==(1,1):continue
                if mode=='empty':continue
                if mode=='three-cells' and row!=0:continue
                if mode=='merge-h' and row==0 and col==1:continue
                if mode=='merge-v' and row==1 and col==0:continue
                left,top=col*500,row*300;right,bottom=left+500,top+300
                if mode=='merge-h'and(row,col)==(0,0):right+=500
                if mode=='merge-v'and(row,col)==(0,0):bottom+=300
                if mode=='jitter':left+=rng.randrange(-2,3);right+=rng.randrange(-2,3);top+=rng.randrange(-2,3);bottom+=rng.randrange(-2,3)
                kind=20 if mode=='line'and row==0 else(2 if mode=='wrong-shape'else 1)
                text=rec(0xf00d,rec(0xfa8,('cell '+str(row)+str(col)).encode()),15)
                leaf=rec(0xf004,rec(0xf00a,b'\0'*8,2,kind)+rec(0xf00f,struct.pack('<4i',left,top,right,bottom))+text,15);leaves.append(leaf)
                if mode=='overlap'and(row,col)==(0,0):leaves.append(leaf)
        group=rec(0xf004,rec(0xf009,struct.pack('<4i',0,0,1500,900))+rec(0xf00f,struct.pack('<4i',0,0,1500,900))+rec(0xf00b,struct.pack('<HI',0x39f,int(mode!='unmarked')),3,1),15)
        add('table/'+mode,rec(1006,rec(0xf003,group+b''.join(leaves),15),15))
    for i in range(200):
        payload=rng.randbytes(rng.randrange(0,100));instance=rng.randrange(0,20);record=r.PptRecord(0,3,instance,0xf00b,payload)
        props.append(dict(payload=payload.hex(),instance=instance,simple=p._fopt_properties(record),complex={k:v.hex()for k,v in p._fopt_complex_properties(record).items()}))
    for size in range(0,30):
        payload=struct.pack('<HIHIHI',0x8004,size,0x0004,99,0x8005,size)+b'ab'*30
        for cut in [len(payload),18,19+size,20+size*2]:
            data=payload[:cut];record=r.PptRecord(0,3,3,0xf00b,data)
            props.append(dict(payload=data.hex(),instance=3,simple=p._fopt_properties(record),complex={k:v.hex()for k,v in p._fopt_complex_properties(record).items()}))
    for mask in [0,0x800000,0x1000000,0x2000000,0x3800000,0xffffffff]:
        for enabled in [-1,0,1,2]:
            for si in [0,1,2,4,32,64,512,0xffffffff]:
                payload=struct.pack('<I',mask)+(b'xx'if mask&0x800000 else b'')+(struct.pack('<h',enabled)if mask&0x2000000 else b'')+(struct.pack('<HH',3,5)if mask&0x1000000 else b'')+struct.pack('<I',0)+struct.pack('<I',si)+b'\0'*20
                for cut in [len(payload),len(payload)-1,8]:
                    b=payload[:cut];numbers.append(dict(data=b.hex(),expected={k:asdict(v)for k,v in p._parse_style_text9_numbering(b).items()}))
    for path in [*sorted((ROOT/'demo').rglob('*.ppt')),ROOT/'tests/fixtures/docvortex_migration/pptx_01.ppt',ROOT/'tests/migration/inputs/synthetic.ppt']:
        if not path.exists():continue
        with BoundedOleReader(path.read_bytes())as ole:
            data=ole.read_stream('PowerPoint Document');current=ole.read_stream('Current User',required=False)
        budget=r.RecordBudget();layout=p._locate_document(data,current,budget);masters,fallback=p._collect_masters(layout,data,budget);external=p._external_text_records(layout.document,budget);links=p._hyperlink_targets(layout.document,budget)
        for reference,slide_id in p._slide_entries(layout.document,budget):
            offset=layout.persist.get(reference)
            if offset is None:continue
            slide=r.record_at(data,offset,budget=budget)
            if slide is None or slide.record_type!=1006:continue
            master=masters.get(p._slide_master_id(slide,budget)or-1,fallback);styles=master[1]if master else{}
            texts=p._parse_text_contents(external.get(reference,[]),styles,links,budget)
            add(path.name+'/'+str(slide_id),rec(slide.record_type,slide.payload,slide.version,slide.instance),styles,texts,links)
    target=ROOT/'tests/fixtures/office_rust_reference_068/ppt_shapes.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,properties=props,numbering=numbers),ensure_ascii=True).encode(),mtime=0))
    print(len(cases),'PPT slide/group checkpoints;',len(props),'FOPT;',len(numbers),'numbering inputs')
if __name__=='__main__':main()
