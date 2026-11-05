"""Original DOCX package repair/CRC failure decisions, not runtime code."""
import gzip,json,struct,sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
NS='http://schemas.openxmlformats.org/package/2006/relationships'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
def rels(rows):return ('<Relationships xmlns="'+NS+'">'+''.join('<Relationship Id="'+id+'" Type="'+R+kind+'" Target="'+target+'"'+(' TargetMode="'+mode+'"' if mode else '')+'/>' for id,kind,target,mode in rows)+'</Relationships>').encode()
def package(parts,corrupt):
    out=BytesIO()
    with ZipFile(out,'w',ZIP_STORED) as z:
        for name,data in parts.items():z.writestr(name,data)
    data=bytearray(out.getvalue())
    with ZipFile(BytesIO(data)) as z:
        for name in corrupt:
            info=z.getinfo(name);offset=info.header_offset;length,extra=struct.unpack_from('<HH',data,offset+26)
            data[offset+30+length+extra]^=0x40
    return bytes(data)
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.package_normalizer import normalize_docx_package
    from loguru import logger
    logger.remove();cases=[]
    base={'[Content_Types].xml':b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>',
        '_rels/.rels':rels([('main','officeDocument','word/document.xml','')]),
        'word/document.xml':b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>body retained</w:t></w:r></w:p></w:body></w:document>',
        'word/_rels/document.xml.rels':rels([('style','styles','styles.xml',''),('image','image','media/image.emf',''),('embed','oleObject','embeddings/object.bin',''),('missing','image','media/missing.png',''),('escape','image','../../outside.xml',''),('external','image','https://example.invalid/image.png','External')]),
        'word/styles.xml':b'<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
        'word/media/image.emf':b'not a metafile', 'word/embeddings/object.bin':b'not OLE', 'orphan.bin':b'orphan payload'}
    def capture(name,parts,corrupt=()):
        data=package(parts,corrupt);case=dict(name=name,hex=data.hex())
        try:
            normalized=normalize_docx_package(data)
            with ZipFile(BytesIO(normalized)) as z:
                case['members']=sorted(z.namelist());case['rels']={}
                from lxml import etree
                ct=etree.fromstring(z.read('[Content_Types].xml'));case['overrides']={n.get('PartName','').replace('\\','/').lstrip('/'):n.get('ContentType','') for n in ct if isinstance(n.tag,str) and n.tag=='{http://schemas.openxmlformats.org/package/2006/content-types}Override'}
                from lxml import etree
                for part in z.namelist():
                    if part.endswith('.rels'):
                        try:case['rels'][part]=[n.get('Id','') for n in etree.fromstring(z.read(part)) if isinstance(n.tag,str) and etree.QName(n).localname=='Relationship']
                        except etree.XMLSyntaxError:pass
        except Exception as e:case['error']=type(e).__name__
        cases.append(case)
    for manifest in [b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',b'<t:Types xmlns:t="http://schemas.openxmlformats.org/package/2006/content-types"><t:Override PartName="/word/document.xml" ContentType="wrong" custom="keep"/></t:Types>',b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml"/></Types>']:
        parts=dict(base);parts['[Content_Types].xml']=manifest;capture('content types '+str(len(cases)),parts)
    for encoding in ['utf-16','utf-16be','windows-1252']:
        parts=dict(base)
        for name in ['[Content_Types].xml','word/_rels/document.xml.rels']:
            text=parts[name].decode();parts[name]=('<?xml version="1.0" encoding="'+encoding+'"?>'+text).encode(encoding)
        capture('encoded '+encoding,parts)
    capture('missing relationship removal',base)
    for name in base:capture('CRC '+name,base,[name])
    for name in ['word/media/image.emf','word/embeddings/object.bin','orphan.bin','word/styles.xml']:
        parts=dict(base);parts['word/_rels/document.xml.rels']=b'<Relationships broken'
        capture('incomplete graph '+name,parts,[name])
    for target in ['media/image.emf','media\\image.emf','/word/media/image.emf','./media/image.emf','media/../media/image.emf','../word/media/image.emf','..','../..','','../../oops','not%20there']:
        parts=dict(base);parts['word/_rels/document.xml.rels']=rels([('r','image',target,'')]);capture('target '+target,parts)
    parts=dict(base);parts['word/media/_rels/image.emf.rels']=rels([('again','image','../image.emf','')]);capture('cycle',parts)
    parts=dict(base);parts['word/embeddings/nested.bin']=b'nested';parts['word/embeddings/_rels/nested.bin.rels']=rels([('back','officeDocument','../document.xml','')]);capture('unreachable relationship XML CRC',parts,['word/embeddings/_rels/nested.bin.rels'])
    (ROOT/'tests/fixtures/office_rust_reference_068/docx_normalize.json.gz').write_bytes(gzip.compress(json.dumps(dict(cases=cases)).encode(),mtime=0));print('DOCX normalization references:',len(cases),'packages;',sum('error' in c for c in cases),'fatal corruptions')
if __name__=='__main__':main()
