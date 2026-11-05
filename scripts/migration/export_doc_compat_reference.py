"""Freeze CPython DOC encodings, CLX recovery and ObjectPool naming behavior."""
import gzip
import json
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc import pieces as py_pieces
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.fib import parse_fib
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.parser import parse_doc_document
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.records import DocBudget
from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.embedded_chart import extract_embedded_chart_html
from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
from rapid_doc.version import __version__


def ints(values):return struct.pack('<'+'I'*len(values),*values)
def doc(payload,*,compressed=True,lid=0x409,far_east=0,clx=False,ranges=None,total=None,complex=False,overflow=False,object_id=None):
    word=bytearray(2048);table=bytearray();pairs=[(0,0)]*93
    struct.pack_into('<HH',word,0,0xa5ec,0xc1)
    flags=0x200 | (4 if complex or clx else 0) | (0x4000 if far_east else 0)
    struct.pack_into('<HH',word,6,lid,0);struct.pack_into('<H',word,10,flags)
    struct.pack_into('<II',word,24,2048,2048+len(payload));struct.pack_into('<H',word,0x3c,far_east)
    length=len(payload)//(1 if compressed else 2)
    struct.pack_into('<I',word,0x4c,length if total is None else total)
    if overflow:struct.pack_into('<II',word,0x50,0xfffffff0,64)
    word.extend(payload)
    def add(index,data):pairs[index]=(len(table),len(data));table.extend(data)
    if clx:
        specs=ranges if ranges is not None else [(0,length,2048,compressed)]
        cps=[r[0] for r in specs]+([specs[-1][1]] if specs else [0])
        pcd=ints(cps)+b''.join(struct.pack('<HIH',0,(0x40000000|fc*2) if ansi else fc,0) for _,_,fc,ansi in specs)
        add(33,b'\x02'+struct.pack('<I',len(pcd))+pcd)
    if object_id is not None:
        text=payload.decode('utf-16le')
        separators=[2048+len(text[:i].encode('utf-16le')) for i,c in enumerate(text) if c=='\x14']
        bounds=sorted({2048,2048+len(payload),*[v for loc in separators for v in (loc,loc+2)]})
        fkp=bytearray(512);fkp[:4*len(bounds)]=ints(bounds)
        for i,start in enumerate(bounds[:-1]):
            if start in separators:fkp[4*len(bounds)+i]=200
        grp=struct.pack('<HI',0x6a03,object_id);fkp[400]=len(grp);fkp[401:407]=grp;fkp[511]=len(bounds)-1
        offset=((len(word)+511)//512)*512;word.extend(b'\0'*(offset-len(word)));word.extend(fkp)
        add(12,ints([bounds[0],bounds[-1],offset//512]))
    for i,pair in enumerate(pairs):struct.pack_into('<II',word,0x9a+i*8,*pair)
    return bytes(word),bytes(table)


def main():
    directory=ROOT/'tests/fixtures/office_rust_reference_068'
    mapped=[]
    lids=[0x409,0x405,0x419,0x408,0x41f,0x40d,0x401,0x425,0x42a,0x41e,0x411,0x804,0x412,0x404]
    for lid in lids:
        codec=py_pieces.codec_for_lid(lid)
        payload=bytes(range(256))
        if codec in ('cp932','cp936','cp949','cp950'):
            leads=[v for v in range(256) if py_pieces._lead_byte(codec,v)]
            payload+=b''.join(bytes([a,b]) for a in leads for b in range(256))
        pieces=[py_pieces.Piece(0,len(payload),0,True)]
        text=py_pieces.extract_text(payload,pieces,total_cp=len(payload),codec=codec,budget=DocBudget())
        mapped.append(dict(lid=lid,hex=payload.hex(),compressed=True,chars=text.chars,cps=text.cps,fcs=text.fcs,piece_indexes=text.piece_indexes))
    payload=struct.pack('<10H',0x41,0xd83d,0xde00,0xd800,0x20,0xdc00,0xfffd,0x42,0xdfff,0xd800)
    text=py_pieces.extract_text(payload,[py_pieces.Piece(0,10,0,False)],total_cp=10,codec='cp1252',budget=DocBudget())
    mapped.append(dict(lid=0x409,hex=payload.hex(),compressed=False,chars=text.chars,cps=text.cps,fcs=text.fcs,piece_indexes=text.piece_indexes))
    (directory/'doc_encoding.json.gz').write_bytes(gzip.compress(json.dumps(mapped).encode(),mtime=0))
    sources=[]
    def add(name,streams,pool=None):sources.append((name,*streams,pool or {}))
    for codec,lid in [('cp932',0x411),('cp936',0x804),('cp949',0x412),('cp950',0x404)]:
        text={'cp932':'日本語','cp936':'简体中文','cp949':'한국어','cp950':'繁體中文'}[codec]+'\r'
        add('far-east-'+codec,doc(text.encode(codec),far_east=lid))
    add('undefined-ansi',doc(b'hello \x81\x8d\x9d\r'))
    word,_=doc(b'recovery must not hide bad CLX\r',complex=True)
    add('complex-without-clx',(word,b'nonempty Table stream without a CLX'))
    word,table=doc(b'abc');word=bytearray(word);struct.pack_into('<I',word,28,2048)
    add('no-recoverable-pieces',(bytes(word),table))
    add('empty-clx',doc(b'',clx=True,ranges=[]))
    add('empty-document',doc(b''))
    add('empty-clx-with-text',doc(b'abc',clx=True,ranges=[]))
    add('decreasing-pieces',doc(b'abcdef',clx=True,ranges=[(0,4,2048,True),(4,2,2052,True)]))
    add('story-count-overflow',doc(b'abc',clx=True,overflow=True))
    field='Chart \x13EMBED Excel.Chart.8\x14\x01\x15\r'.encode('utf-16le')
    chart_doc=doc(field,compressed=False,clx=True,object_id=100)
    excel=(directory/'doc-object-0.bin').read_bytes();graph=(directory/'graph-all.bin').read_bytes()
    add('chart-numeric-alias',chart_doc,{'ObjectPool/_00100/Workbook':excel,'ObjectPool/_100/Workbook':graph})
    add('chart-invalid-alias-first',chart_doc,{'ObjectPool/_00100/Workbook':b'broken','ObjectPool/_100/Workbook':graph})
    add('chart-invalid-storage-names',chart_doc,{'ObjectPool/100/Workbook':excel,'ObjectPool/__100/Book':excel,'ObjectPool/_100/nested/Workbook':excel})
    references=[]
    class Writer:
        def write(self,name,data):
            raise AssertionError('These chart fixtures must not contain bitmap assets')
    pattern=re.compile(r'^ObjectPool/_([0-9]+)/(?:Workbook|Book)$',re.I)
    for name,word,table,pool in sources:
        c=dict(name=name,word_hex=word.hex(),table_hex=table.hex(),pool={k:v.hex() for k,v in pool.items()})
        try:
            charts={}
            for path,data in sorted(pool.items()):
                match=pattern.match(path)
                if match and int(match[1]) not in charts:
                    html=extract_embedded_chart_html(data)
                    if html:charts[int(match[1])]=html
            pages=DocConverter._document_pages(parse_doc_document(word,table,b'',parse_fib(word),native_charts=charts))
            middle=result_to_middle_json(pages,Writer())
            c.update(pages=pages,middle=middle,exports={mode:union_make(middle['pdf_info'],mode,'images') for mode in
                     ['mm_markdown','nlp_markdown','content_list','content_list_v2']})
        except Exception as error:c['error']=type(error).__name__
        references.append(c)
    (directory/'doc_compat.json.gz').write_bytes(gzip.compress(json.dumps(dict(version=__version__,cases=references)).encode(),mtime=0))
    print('DOC codec mappings:',len(mapped),'body/object cases:',len(references))


if __name__=='__main__':main()
