"""Generate genuine Word97 streams and freeze Python note/story contracts."""
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


def cp(text):return len(text.encode('utf-16le'))//2
def ints(values):return struct.pack('<'+'I'*len(values),*values)


def streams(name):
    main='First \x02😀 footnote.\rSecond \x02 and end \x02.\rFigure \x13 SEQ Figure \\* ARABIC \x141\x15 caption.\r'
    positions=[cp(main[:i]) for i,c in enumerate(main) if c=='\x02']
    notes=['\x02\tAlpha footnote.\rcontinued.\r','\x02\x13 HYPERLINK "https://example.com" \x14Beta\x15 footnote.\r']
    if name=='empty-note':notes[0]='\r'
    if name=='unterminated-field':notes[0]='\x02\x13 HYPERLINK "https://bad.example" \x14open result\r'
    end='\x02Endnote content.\r'
    header='\x13 HYPERLINK "https://header.example" \x14Header\x15\r'
    native=b''
    if name=='native-objects':
        sys.path.insert(0,str(ROOT/'tests/native_office'))
        from _mtef_test_utils import equation_native,mtef_equation,mtef_char
        native=equation_native(mtef_equation(mtef_char('x')))
        formula='\x13EMBED Equation.3\x14\x01\x15'
        main+='Math '+formula+'.\r'
        notes[0]='\x02Note '+formula+'.\r'
        header='Header '+formula+'\r'
        end='\x02End '+formula+'\r'
    textbox='TEXTBOX OMITTED BY PYTHON\r'
    counts=[cp(main),cp(''.join(notes)),cp(header),0,0,cp(end),cp(textbox),0]
    all_text=main+''.join(notes)+header+end+textbox
    table=bytearray();pairs=[(0,0)]*93
    def add(index,data):
        pairs[index]=(len(table),len(data));table.extend(data)
    pcd=ints([0,cp(all_text)])+struct.pack('<HIH',0,2048,0)
    add(33,b'\x02'+struct.pack('<I',len(pcd))+pcd)
    split=cp(main[:main.index('\r')+1])
    add(6,ints([0,split,cp(main)])+b'\0'*24)
    references=positions[:2]
    if name=='duplicate-reference':references[1]=references[0]
    add(2,ints(references+[cp(main)])+b'\0'*4)
    note_text=ints([0,cp(notes[0]),cp(''.join(notes))])
    if name=='truncated-note-plc':note_text=note_text[:-1]
    add(3,note_text)
    add(46,ints([positions[2],cp(main)])+b'\0'*2)
    add(47,ints([0,cp(end)]))
    add(11,ints([0]*7+[cp(header)]*12))
    if name=='code-style':
        main='  code\x0bline 2\r';all_text=main;counts=[cp(main)]+[0]*7
        table=bytearray();pairs=[(0,0)]*93
        pcd=ints([0,cp(main)])+struct.pack('<HIH',0,2048,0)
        add(33,b'\x02'+struct.pack('<I',len(pcd))+pcd)
        raw=struct.pack('<HHH',0,0xfff0,0)+b'\0'*4+struct.pack('<H',4)+'Code'.encode('utf-16le')+b'\0\0'
        stsh=bytearray(20);struct.pack_into('<HHH',stsh,0,18,1,10)
        add(1,stsh+struct.pack('<H',len(raw))+raw)
    word=bytearray(2048);struct.pack_into('<HH',word,0,0xa5ec,0xc1)
    struct.pack_into('<H',word,6,0x409);struct.pack_into('<H',word,10,0x204)
    struct.pack_into('<II',word,24,2048,2048+len(all_text.encode('utf-16le')))
    cursor=32
    struct.pack_into('<HH',word,cursor,0,11);cursor+=4
    rglw=[0]*11;rglw[3:11]=counts
    struct.pack_into('<11I',word,cursor,*rglw);cursor+=44
    struct.pack_into('<H',word,cursor,len(pairs));cursor+=2
    for offset,size in pairs:struct.pack_into('<II',word,cursor,offset,size);cursor+=8
    struct.pack_into('<H',word,cursor,0)
    word.extend(all_text.encode('utf-16le'))
    if native:
        locations=[2048+2*cp(all_text[:i]) for i,c in enumerate(all_text) if c=='\x14']
        bounds=sorted({2048,2048+2*cp(all_text),*[v for loc in locations for v in (loc,loc+2)]})
        count=len(bounds)-1;fkp=bytearray(512)
        fkp[:4*len(bounds)]=ints(bounds)
        for i,start in enumerate(bounds[:-1]):
            if start in locations:fkp[4*len(bounds)+i]=200
        grp=struct.pack('<HI',0x6a03,100)
        fkp[400]=len(grp);fkp[401:401+len(grp)]=grp;fkp[511]=count
        offset=((max(4096,len(word))+511)//512)*512
        word.extend(b'\0'*(offset-len(word)));word.extend(fkp)
        add(12,ints([bounds[0],bounds[-1],offset//512]))
        pair_start=32+4+44+2
        struct.pack_into('<II',word,pair_start+12*8,*pairs[12])
    return bytes(word),bytes(table),native


def main():
    cases=[]
    for name in ['basic','empty-note','unterminated-field','duplicate-reference','truncated-note-plc','code-style','native-objects']:
        word,table,native=streams(name)
        document=parse_doc_document(word,table,b'',parse_fib(word),native_equations={100:'x'} if native else None)
        pages=DocConverter._document_pages(document)
        middle=result_to_middle_json(pages,None)
        cases.append(dict(name=name,word_hex=word.hex(),table_hex=table.hex(),native_hex=native.hex(),pages=pages,middle=middle,
            exports={mode:union_make(middle['pdf_info'],mode,'images') for mode in
                     ['mm_markdown','nlp_markdown','content_list','content_list_v2']}))
    path=ROOT/'tests/fixtures/office_rust_reference_068/doc_stories.json'
    path.write_text(json.dumps(dict(version=__version__,cases=cases),ensure_ascii=False),encoding='utf-8')
    print('DOC stream references:',len(cases))


if __name__=='__main__':main()
