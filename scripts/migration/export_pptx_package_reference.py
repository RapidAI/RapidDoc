"""Original PPTX normalizer member semantics and malformed-package boundaries."""
import base64,gzip,hashlib,json,sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from lxml import etree
    from loguru import logger
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.package_normalizer import normalize_pptx_package,STRICT_OOXML_REPLACEMENTS
    logger.remove();base=ROOT/'tests/fixtures/office_rust_reference_068';context=json.loads(gzip.decompress((base/'pptx_context.json.gz').read_bytes()))
    raw=base64.b64decode(next(c for c in context['cases']if c['name']=='generated/base')['package_b64'])
    with ZipFile(BytesIO(raw))as z:parts={n:z.read(n)for n in z.namelist()}
    cases=[]
    def canonical(n):
        name=lambda s: [etree.QName(s).namespace or '',etree.QName(s).localname]
        text=lambda s: s if s and (s.strip() or n.tag.endswith('}t')) else ''
        return [name(n.tag),sorted([[name(k),v]for k,v in n.attrib.items()]),text(n.text),[[canonical(c),c.tail if c.tail and c.tail.strip()else'']for c in n if isinstance(c.tag,str)]]
    def add(name,raw):
        c=dict(name=name,package_b64=base64.b64encode(raw).decode())
        try:
            normalized=normalize_pptx_package(raw);members={}
            with ZipFile(BytesIO(normalized))as z:
                for n in z.namelist():
                    data=z.read(n)
                    try:members[n]=dict(xml=canonical(etree.fromstring(data)))
                    except Exception:members[n]=dict(sha256=hashlib.sha256(data).hexdigest())
            c.update(changed=normalized!=raw,members=members)
        except Exception as e:c['error']=type(e).__name__
        from rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.pptx_converter import PptxConverter
        try:
            engine=PptxConverter();engine.convert(BytesIO(raw));c['pages']=engine.pages
        except Exception as e:c['conversion_error']=type(e).__name__
        cases.append(c)
    def pack(items):
        out=BytesIO()
        with ZipFile(out,'w',ZIP_STORED)as z:
            for n,b in items.items():z.writestr(n,b)
        return out.getvalue()
    def variant(name,change):
        items=dict(parts);change(items);add(name,pack(items))
    add('base',raw)
    for path in sorted((ROOT/'tests/fixtures/pptx').glob('*.pptx')):add(path.name,path.read_bytes())
    def strict(items):
        for n,b in list(items.items()):
            if n.endswith(('.xml','.rels')):
                for s,t in reversed(STRICT_OOXML_REPLACEMENTS):b=b.replace(t,s)
                items[n]=b
    variant('strict',strict)
    for filename in ['ppt/presentation.xml','ppt/slides/slide1.xml','ppt/notesSlides/notesSlide1.xml']:
        def missing(items,filename=filename):
            b=items[filename];end=b.find(b'>',b.find(b'<',b.find(b'?>')+2));items[filename]=b[:end]+b' custom="w:missing"'+b[end:]
        variant('missing-w/'+filename,missing)
    for action in ['missing-overrides','wrong-types','no-content-types','invalid-content-types','invalid-rels']:
        def change(items,action=action):
            path='[Content_Types].xml'
            if action=='no-content-types':items.pop(path)
            elif action=='invalid-content-types':items[path]=b'<Types'
            elif action=='invalid-rels':items['ppt/_rels/presentation.xml.rels']=b'<Relationships'
            else:
                root=etree.fromstring(items[path])
                for n in list(root):
                    if n.tag.endswith('}Override'):
                        if action=='missing-overrides':root.remove(n)
                        else:n.set('ContentType','application/xml')
                items[path]=etree.tostring(root)
        variant(action,change)
    for fallback in [0,1,2]:
        def change(items,fallback=fallback):
            root=etree.fromstring(items['ppt/slides/slide1.xml']);ns=root.nsmap
            tree=root.find('{'+ns['p']+'}cSld/{'+ns['p']+'}spTree');shape=tree.find('{'+ns['p']+'}sp')
            mc='http://schemas.openxmlformats.org/markup-compatibility/2006';alt=etree.SubElement(tree,'{'+mc+'}AlternateContent')
            choice=etree.SubElement(alt,'{'+mc+'}Choice');etree.SubElement(choice,'{'+ns['p']+'}contentPart')
            fb=etree.SubElement(alt,'{'+mc+'}Fallback')
            import copy
            for i in range(fallback):fb.append(copy.deepcopy(shape))
            items['ppt/slides/slide1.xml']=etree.tostring(root)
        variant('content-part/'+str(fallback),change)
    for name in [next(n for n in parts if n.startswith('ppt/media/')),'ppt/media/corrupt.png','extra/corrupt.bin','ppt/embeddings/corrupt.bin']:
        items=dict(parts);items[name]=b'CRC-SENTINEL-CONTENT';blob=bytearray(pack(items));offset=blob.index(b'CRC-SENTINEL-CONTENT');blob[offset]^=1;add('bad-crc/'+name,bytes(blob))
    for value in ['http://purl.oclc.org/ooxml/officeDocument/relationships/metadata/thumbnail','http://purl.oclc.org/ooxml/officeDocument/relationships/customProperties','http://purl.oclc.org/ooxml/officeDocument/relationships/extendedProperties']:
        variant('long-uri/'+value.rsplit('/',1)[-1],lambda items,value=value:items.update({'extra.xml':('<extra value="'+value+'"/>').encode()}))
    (base/'pptx_package.json.gz').write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print(len(cases),'PPTX normalizer cases;',sum('error'in c for c in cases),'original errors')
if __name__=='__main__':main()
