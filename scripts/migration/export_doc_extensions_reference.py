"""DOC textbox/comment/code binary corpus with independent semantic checks."""
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.fib import parse_fib
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.parser import parse_doc_document
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
from rapid_doc.version import __version__

def cp(s):return len(s.encode('utf-16le'))//2
def ints(v):return struct.pack('<'+'I'*len(v),*v)
def build(name):
    main='Body 😀 \x05 comment.\rSecond section.\r'
    header='Header\rFooter\r'
    annotation='Review \x13HYPERLINK "https://example.org"\x14this\x15.\r'
    textbox='Box α.\rLinked continuation.\rSPARE\r'
    if name=='textbox-table':textbox='Cell A\x07Cell B\x07\rSPARE\r'
    headerbox='Header box.\rFooter box.\rSPARE\r'
    code=name.startswith('code')
    if code:
        main={'code-literal':'  print(```)\x0b\treturn <&>\r','code-blank':'  \t\r','code-four-ticks':'````\x0bline\r'}[name]
        header=annotation=textbox=headerbox=''
    texts=[main,'',header,'',annotation,'',textbox,headerbox]
    all_text=''.join(texts);counts=[cp(s) for s in texts]
    table=bytearray();pairs=[(0,0)]*93
    def add(i,d):pairs[i]=(len(table),len(d));table.extend(d)
    pcd=ints([0,cp(all_text)])+struct.pack('<HIH',0,2048,0)
    add(33,b'\x02'+struct.pack('<I',len(pcd))+pcd)
    if not code:
        split=cp(main[:main.index('\r')+1])
        add(6,ints([0,split,cp(main)])+b'\0'*24)
        # Two used textboxes and a mandatory spare entry. A linked chain is read once.
        def ftx(lid,reuse=0,chain=1):return struct.pack('<iiHiii',chain,0,reuse,0,lid,0)
        def spa(lid):return struct.pack('<I',lid)+b'\0'*22
        end=cp(textbox[:textbox.index('SPARE')])
        entries=[ftx(100,0,2),ftx(999,1),ftx(0,0)]
        add(56,ints([0,end,end+1,cp(textbox)])+b''.join(entries))
        anchor=2
        if name=='textbox-unbound':anchor_id=404
        else:anchor_id=100
        add(40,ints([anchor,cp(main)])+spa(anchor_id))
        a=cp('Header box.\r');b=cp('Header box.\rFooter box.\r')
        add(58,ints([0,a,b,cp(headerbox)])+ftx(200)+ftx(201)+ftx(0))
        add(41,ints([0,cp('Header\r'),cp(header)])+spa(200)+spa(201))
        # Header and footer in the first section, none in the second.
        h=cp('Header\r');n=cp(header)
        add(11,ints([0]*7+[h,h,n,n,n]+[n]*7))
        at=cp(main[:main.index('\x05')])
        references=[at]
        if name=='comment-duplicate':references=[at,at]
        if name=='comment-outside':references=[cp(main)+4]
        descriptor=struct.pack('<H',2)+'AB'.encode('utf-16le')+b'\0'*24
        add(4,ints(references+[cp(main)])+descriptor*len(references))
        ends=[0,cp(annotation)]
        if name=='comment-duplicate':ends.append(cp(annotation))
        if name=='comment-reversed':ends=[cp(annotation),0]
        if name=='comment-outside':ends=[0,cp(annotation)+500]
        add(5,ints(ends))
        if name=='textbox-truncated':
            offset,size=pairs[56];pairs[56]=(offset,size-1)
    else:
        raw=struct.pack('<HHH',0,0xfff0,0)+b'\0'*4+struct.pack('<H',4)+'Code'.encode('utf-16le')+b'\0\0'
        stsh=bytearray(20);struct.pack_into('<HHH',stsh,0,18,1,10)
        add(1,stsh+struct.pack('<H',len(raw))+raw)
    word=bytearray(2048)
    struct.pack_into('<HH',word,0,0xa5ec,0xc1);struct.pack_into('<H',word,6,0x409);struct.pack_into('<H',word,10,0x204)
    struct.pack_into('<II',word,24,2048,2048+len(all_text.encode('utf-16le')))
    struct.pack_into('<HH',word,32,0,11);struct.pack_into('<11I',word,36,0,0,0,*counts)
    struct.pack_into('<H',word,80,len(pairs))
    for i,pair in enumerate(pairs):struct.pack_into('<II',word,82+8*i,*pair)
    word.extend(all_text.encode('utf-16le'))
    return bytes(word),bytes(table)

def plain(v):
    if isinstance(v,str):return v
    if isinstance(v,list):return ''.join(plain(x) for x in v)
    if isinstance(v,dict):return plain(v.get('content',''))
    return ''

def main():
    cases=[]
    for name in ['linked-textboxes-comments','textbox-table','textbox-unbound','textbox-truncated','comment-duplicate','comment-reversed','comment-outside','code-literal','code-blank','code-four-ticks']:
        word,table=build(name)
        pages=DocConverter._document_pages(parse_doc_document(word,table,b'',parse_fib(word)))
        middle=result_to_middle_json(pages,None)
        exports={mode:union_make(middle['pdf_info'],mode,'images') for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']}
        visible=plain([b for p in pages for b in p])
        assert 'SPARE' not in visible,name
        if name=='linked-textboxes-comments':
            assert visible.count('Linked continuation.')==1
            assert any(b['type']=='page_footnote' and plain(b).startswith('[comment 1] Review') for b in pages[0])
            assert all(b['type']!='page_footnote' for b in pages[1])
            assert any(b['type']=='header' and plain(b)=='Header box.' for b in pages[0])
            assert any(b['type']=='footer' and plain(b)=='Footer box.' for b in pages[0])
        if name=='textbox-table':
            assert any(b['type']=='table' and 'Cell A' in b['content'] and 'Cell B' in b['content'] for b in pages[0])
        if name.startswith('code'):
            assert pages[0][0]['type']=='code'
            expected=build(name)[0][2048:].decode('utf-16le').rstrip('\r').replace('\x0b','\n')
            assert middle['pdf_info'][0]['para_blocks'][0]['lines'][0]['spans'][0]['content']==expected
            assert exports['content_list'][0]['text']==expected
            assert exports['content_list_v2'][0][0]['content']['code_content']==expected
            assert expected in exports['mm_markdown']
        cases.append(dict(name=name,word_hex=word.hex(),table_hex=table.hex(),pages=pages,middle=middle,exports=exports))
    destination=ROOT/'tests/fixtures/office_rust_reference_068/doc_extensions.json'
    destination.write_text(json.dumps(dict(version=__version__,cases=cases),ensure_ascii=False),encoding='utf8')
    print('DOC extension cases:',len(cases))

if __name__=='__main__':main()
