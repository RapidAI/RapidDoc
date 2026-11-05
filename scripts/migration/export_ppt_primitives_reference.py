"""Freeze existing PPT record, save-chain, style and UTF-16 text behavior."""
import gzip,json,random,struct,sys,unicodedata
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt import records as r,style_text as s,parser as p
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeResourceLimitError,LegacyOfficeMalformedError
    from loguru import logger
    logger.remove();rng=random.Random(76549);records=[];styles=[];texts=[];documents=[];parses=[]
    controls=[];prepared=[];contexts=[]
    for code in range(0x110000):
        if unicodedata.category(chr(code)).startswith('C'):
            if controls and controls[-1][1]+1==code:controls[-1][1]=code
            else:controls.append([code,code])
    (ROOT/'rust/rapiddoc-office/src/ppt/control_categories.json').write_text(json.dumps(controls)+'\n',encoding='utf-8')
    def rec(kind,payload=b'',version=0,instance=0):return struct.pack('<HHI',(instance<<4)|version,kind,len(payload))+payload
    def error(e):return 'resource_limit'if isinstance(e,LegacyOfficeResourceLimitError)else'malformed'
    def summary(record):return dict(offset=record.offset,version=record.version,instance=record.instance,kind=record.record_type,payload=record.payload.hex())
    def add_records(name,data):
        for strict in [False,True]:
            b=r.RecordBudget();c=dict(name=name,data=data.hex(),strict=strict)
            try:c['records']=[summary(x)for x in r.iter_records(data,budget=b,strict_first=strict)]
            except Exception as e:c['error']=error(e)
            c['count']=b.count
            try:
                b=r.RecordBudget();root=r.PptRecord(0,15,0,1000,data);c['descendants']=[summary(x)for x in r.iter_descendants(root,budget=b)];c['descendant_count']=b.count
            except Exception as e:c['descendant_error']=error(e)
            records.append(c)
    for i in range(80):
        blob=b''
        for j in range(rng.randrange(0,12)):
            leaf=rec(rng.choice([4000,4008,1001,0xf004]),rng.randbytes(rng.randrange(0,25)),instance=rng.randrange(0,20))
            for _ in range(rng.randrange(0,5)):leaf=rec(rng.choice([1052,1053,1064,1065,1000]),leaf,15)
            blob+=leaf
        add_records('random/'+str(i),blob)
        if blob:add_records('truncated/'+str(i),blob[:-rng.randrange(1,min(8,len(blob))+1)])
    for depth in [63,64,65,66]:
        blob=rec(4008,b'text')
        for _ in range(depth):blob=rec(1000,blob,15)
        add_records('depth/'+str(depth),blob)
    for i in range(300):
        data=rng.randbytes(rng.randrange(0,150));length=rng.randrange(0,70);instance=rng.randrange(0,12)
        styles.append(dict(data=data.hex(),length=length,instance=instance,style=asdict(s.parse_style_text(data,length)),master=[asdict(x)for x in s.parse_master_style(data,instance)]))
    # Structured masks exercise every skip field and its truncation boundary.
    for pf in [0,1,15,0x80,0x10,0x40,0x20,0x800,0x1000,0x2000,0x4000,0x100,0x400,0x8000,0x100000,0x10000,0xe0000,0x200000,0x3fffff]:
        for cf in [0,1,2,4,0x100,0xffff,0x10000,0x200000,0x400000,0x800000,0x20000,0x40000,0x80000,0xffffff]:
            pfbody=struct.pack('<I',pf)+b'\x01\x00'*100;pfparsed=s._parse_paragraph_exception(pfbody,0)
            cfbody=struct.pack('<I',cf)+b'\xff\xff'*100;cfparsed=s._parse_character_exception(cfbody,0)
            body=struct.pack('<IH',41,12)+pfbody[:pfparsed[1]]+struct.pack('<I',41)+cfbody[:cfparsed[1]]
            for cut in [len(body),max(0,len(body)-1),rng.randrange(0,len(body)+1)]:
                b=body[:cut];master=struct.pack('<H',1)+pfbody[:pfparsed[1]]+cfbody[:cfparsed[1]]
                styles.append(dict(data=b.hex(),length=40,instance=0,style=asdict(s.parse_style_text(b,40)),master=[asdict(x)for x in s.parse_master_style(b,0)]))
            for instance in [0,5]:
                master=struct.pack('<H',2)+(b'\0\0'if instance>=5 else b'')+pfbody[:pfparsed[1]]+cfbody[:cfparsed[1]]
                styles.append(dict(data=master.hex(),length=0,instance=instance,style=asdict(s.parse_style_text(master,0)),master=[asdict(x)for x in s.parse_master_style(master,instance)]))
    for i in range(400):
        text=''.join(rng.choice(['a','中','😀','\r','\n','\v','\t',' ','\0','\u200d','\ue000','\u0378'])for _ in range(rng.randrange(0,100)))
        runs=s.StyleRuns(paragraphs=[s.ParagraphRun(count=rng.randrange(0,30),depth=rng.randrange(0,9),bullet=rng.choice([None,True,False]))for _ in range(rng.randrange(0,7))],characters=[s.CharacterRun(count=rng.randrange(0,30),bold=rng.choice([None,True,False]),italic=rng.choice([None,True,False]),underline=rng.choice([None,True,False]),strike=rng.choice([None,True,False]),baseline=rng.choice([None,-15,0,15]),pp9rt=rng.randrange(0,16))for _ in range(rng.randrange(0,7))])
        masters=[s.MasterLevel(bullet=rng.choice([None,True,False]),bold=rng.choice([None,True,False]),italic=rng.choice([None,True,False]),underline=rng.choice([None,True,False]),baseline=rng.choice([None,-10,10]))for _ in range(rng.randrange(0,5))]
        links=[(a,a+rng.randrange(0,50),'https://example.com/'+str(j))for j,a in enumerate([rng.randrange(0,100)for _ in range(3)])]
        texts.append(dict(text=text,styles=asdict(runs),masters=[asdict(x)for x in masters],links=links,paragraphs=[asdict(x)for x in p._build_paragraphs(text,runs,masters,links)]))
    def add_document(name,data,current=b''):
        c=dict(name=name,data=data.hex(),current=current.hex());budget=r.RecordBudget()
        try:
            layout=p._locate_document(data,current,budget);c['expected']=dict(document_offset=layout.document.offset,persist=layout.persist,recovered=layout.recovered,size=p._presentation_size(layout.document,budget),slides=p._slide_entries(layout.document,budget),masters=p._master_entries(layout.document,budget),notes=p._notes_entries(layout.document,budget),record_visits=budget.count)
        except Exception as e:c['error']=error(e)
        documents.append(c)
    document=rec(1000,rec(1001,struct.pack('<II',5760,4320))+rec(0xff0,rec(1011,struct.pack('<IIII',2,0,0,100)),15),15)
    for mode in ['none','valid','self','bad-directory','missing-document','cyclic']:
        data=bytearray(document);directory_offset=len(data);data.extend(rec(0x1772,struct.pack('<III',(2<<20)|1,0,len(document))))
        edit_offset=len(data);edit=bytearray(28);struct.pack_into('<III',edit,8,edit_offset if mode=='self'else 0,directory_offset+3 if mode=='bad-directory'else directory_offset,77 if mode=='missing-document'else 1);data.extend(rec(0xff5,edit))
        current=bytearray(20);struct.pack_into('<I',current,16,edit_offset if mode!='none'else 0);add_document(mode,bytes(data),bytes(current))
    for count in [99,100,101]:
        data=bytearray(document);directory=len(data);data.extend(rec(0x1772,struct.pack('<II',(1<<20)|1,0)));previous=0
        for _ in range(count):
            offset=len(data);edit=bytearray(28);struct.pack_into('<III',edit,8,previous,directory,1);data.extend(rec(0xff5,edit));previous=offset
        current=bytearray(20);struct.pack_into('<I',current,16,previous);add_document('chain/'+str(count),bytes(data),bytes(current))
    paths=[*sorted((ROOT/'demo').rglob('*.ppt')),ROOT/'tests/fixtures/docvortex_migration/pptx_01.ppt',ROOT/'tests/migration/inputs/synthetic.ppt']
    for path in paths:
        if not path.exists():continue
        with BoundedOleReader(path.read_bytes())as ole:
            raw=ole.read_stream('PowerPoint Document');current=ole.read_stream('Current User',required=False)
            add_document(path.relative_to(ROOT).as_posix(),raw,current)
        budget=r.RecordBudget();layout=p._locate_document(raw,current,budget);masters,fallback=p._collect_masters(layout,raw,budget);external=p._external_text_records(layout.document,budget);links=p._hyperlink_targets(layout.document,budget)
        notes,unbound=p._notes_by_slide_id(layout,raw,fallback[1]if fallback else{},links,budget)
        contexts.append(dict(name=path.relative_to(ROOT).as_posix(),data=raw.hex(),current=current.hex(),masters=[dict(id=k,offset=v[0].offset,styles={k:[asdict(x)for x in levels]for k,levels in v[1].items()})for k,v in masters.items()],external={k:[summary(x)for x in v]for k,v in external.items()},links=links,notes={k:[asdict(x)for x in v]for k,v in notes.items()},unbound=[[asdict(x)for x in v]for v in unbound],count=budget.count))
    def add_prepared(name,data,current=b''):
        c=dict(name=name,data=data.hex(),current=current.hex());budget=r.RecordBudget()
        try:
            if current:
                kind=p.get_u16(current,2)
                if kind not in {None,0xff6}:raise LegacyOfficeMalformedError('old ppt')
                if p.get_u32(current,12)==0xf3d1c4df:
                    c['error']='encrypted';prepared.append(c);return
            for root in r.iter_records(data,budget=budget,strict_first=True):
                for _ in r.iter_descendants(root,budget=budget):pass
            layout=p._locate_document(data,current,budget)
            if any(n.record_type==0x2f14 for n in r.iter_descendants(layout.document,budget=budget)):c['error']='encrypted'
            else:c['expected']=dict(document_offset=layout.document.offset,persist=layout.persist,recovered=layout.recovered,count=budget.count)
        except Exception as e:c['error']=error(e)
        prepared.append(c)
    for c in documents:add_prepared(c['name'],bytes.fromhex(c['data']),bytes.fromhex(c['current']))
    for current in [b'x',b'xx',struct.pack('<HH',0,0xff5),struct.pack('<HH',0,0xff6)+b'\0'*8+struct.pack('<I',0xf3d1c4df)]:add_prepared('current/'+current.hex(),document,current)
    add_prepared('crypt-record',rec(1000,rec(0x2f14),15))
    for data in [b'',b'xx',rec(1000,b'xxx',15),struct.pack('<HHI',15,1000,0xffffffff)]:add_prepared('malformed/'+data.hex(),data)
    utf16=[dict(data=bytes(range(256)).hex(),text=r.utf16_text(bytes(range(256))))]
    for _ in range(300):
        data=rng.randbytes(rng.randrange(0,100));utf16.append(dict(data=data.hex(),text=r.utf16_text(data)))
    target=ROOT/'tests/fixtures/office_rust_reference_068/ppt_primitives.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(records=records,styles=styles,texts=texts,documents=documents,utf16=utf16,prepared=prepared,contexts=contexts),ensure_ascii=True).encode(),mtime=0))
    print(len(records),'record streams;',len(styles),'style inputs;',len(texts),'UTF16 paragraph queries;',len(documents),'save layouts;',len(utf16),'UTF16 byte inputs')
if __name__=='__main__':main()
