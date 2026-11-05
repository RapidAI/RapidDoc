"""Freeze old URI rules and actual PDFium link-annotation extraction."""
import json,unicodedata,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    if '--limits-only'in sys.argv:
        folder=ROOT/'build/migration/pdf-link-annotations';folder.mkdir(parents=True,exist_ok=True);records=[]
        for name,target_value,quad in [('uri-limit','https://a.test/'+'x'*1_000_000,''),('quad-limit','https://a.test',' /QuadPoints ['+'20 150 80 150 20 130 80 130 '*5001+']')]:
            path=folder/(name+'.pdf')
            if path.exists():raise FileExistsError(path)
            objs=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Count 1 /Kids [3 0 R] >>',b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 180] /Annots [4 0 R] >>',('<< /Type /Annot /Subtype /Link /Rect [20 130 80 150]'+quad+' /A << /S /URI /URI <'+target_value.encode().hex()+'> >> >>').encode()]
            out=bytearray(b'%PDF-1.7\n');offsets=[0]
            for i,value in enumerate(objs,1):offsets.append(len(out));out.extend(str(i).encode()+b' 0 obj\n'+value+b'\nendobj\n')
            start=len(out);out.extend(b'xref\n0 5\n0000000000 65535 f \n')
            for offset in offsets[1:]:out.extend(f'{offset:010d} 00000 n \n'.encode())
            out.extend(b'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF\n');path.write_bytes(out);records.append(dict(pdf=str(path.relative_to(ROOT)),sha256=digest(out)))
        write_json(folder/'limits.json',dict(scope='authored resource-limit inputs, not Python parity evidence',cases=records));print('Generated two authored PDF URI/quad resource-limit cases');return
    target=ROOT/'build/migration/pdf-link-annotations.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf.native_annotations import _validate_pdf_external_link_target
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    uris=[]
    def uri(s):uris.append(dict(value=s,expected=_validate_pdf_external_link_target(s)))
    seeds=['https://example.test/a?b=1#x',' HTTP://User:Pass@EXAMPLE.TEST:bad/a ','mailto:a@example.test','tel:+123','ftp://a.test','javascript:alert(1)','data:text/a,hello','file:///x','//a.test/x','/x','#x','.', '', 'https:example.test','https:///path','https://','http://:80','http://@','http://a:b:c','mailto://a.test','mailto://a.test/path','tel:?q=x','http://[::1]/a','http://[2001:db8::1]:bad/a','http://[::1%eth0]','http://[vF.name]','http://[VF.name]','http://[127.0.0.1]','http://[bad]','http://[::1]tail','http://host]','http://[host','http://u[a]@foo','http://u[a]@vF.foo','https://例子.测试/路径','http://a\x00b','http://a\tb','http://a<b','http://a\\b']
    for s in seeds:uri(s)
    for c in ['\t','\n','\x00','\x1f','\x7f','\x85','\u00a0','\u2003','\u2028','\u202f','\u3000','＜','＞','／','：','？','＃','＠','℀','℁','℅','℆']:
        for form in [c+'https://a.test'+c,'https://a'+c+'b.test/x','mailto:a'+c+'b','https://a.test/x'+c+'y']:uri(form)
    for host in ['::','::1','2001:db8::1','::ffff:192.0.2.1','::1%zone','::1%','::1%a%b','v1.x','v.a','v1.','v01.a:b','v1.😀','192.0.2.1','0127.0.0.1','1:2:3:4:5:6:7:8','1:2:3:4:5:6:7:8:9']:
        for tail in ['',':80',':bad',']',':bad]']:uri('https://['+host+']'+tail+'/x')
    folder=ROOT/'build/migration/pdf-link-annotations';folder.mkdir(parents=True,exist_ok=True)
    generated=folder/'links.pdf'
    if generated.exists():raise FileExistsError(generated)
    # Raw minimal PDF keeps URI bytes, action types, flags and QuadPoints unmodified by a writer.
    objects=[]
    def obj(value):objects.append(value);return len(objects)
    obj(b'<< /Type /Catalog /Pages 2 0 R >>');obj(b'')
    page_ids=[]
    for rotation in [0,90,180,270]:
        page_id=obj(b'');page_ids.append(page_id)
        text=b'BT /F1 12 Tf 30 130 Td (Linked example text) Tj ET';stream=obj(b'<< /Length '+str(len(text)).encode()+b' >>\nstream\n'+text+b'\nendstream')
        font=obj(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
        annotations=[]
        specs=[(b'https://example.test/a',0,b'',b'/S /URI'),(b'mailto:a@example.test',0,b'/QuadPoints [25 145 80 145 25 128 80 128 90 145 150 145 90 128 150 128 25 145 80 145 25 128 80 128]',b'/S /URI'),(b'https://hidden.test',2,b'',b'/S /URI'),(b'javascript:alert(1)',0,b'',b'/S /URI'),(b'https://noview.test',32,b'',b'/S /URI'),(b'https://invisible.test',1,b'',b'/S /URI'),(b'https://print.test',4,b'',b'/S /URI'),(b'http://[bad]',0,b'',b'/S /URI'),(b'https://safe.test',0,b'/QuadPoints [10 10 10 10 10 10 10 10]',b'/S /URI'),(b'https://bad-utf8.test/\xff',0,b'',b'/S /URI'),(b'https://ignored.test',0,b'',b'/S /GoTo'),(b'  https://trim.test/x  ',0,b'',b'/S /URI'),('https://例子.测试/路径'.encode(),0,b'',b'/S /URI')]
        for value,flags,quad,action in specs:
            a=obj(b'<< /Type /Annot /Subtype /Link /Rect [25 125 160 148] /F '+str(flags).encode()+b' '+quad+b' /A << '+action+b' /URI <'+value.hex().encode()+b'> >> >>');annotations.append(a)
        objects[page_id-1]=b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 180] /CropBox [10 15 230 170] /Rotate '+str(rotation).encode()+b' /Resources << /Font << /F1 '+str(font).encode()+b' 0 R >> >> /Contents '+str(stream).encode()+b' 0 R /Annots ['+b' '.join(str(a).encode()+b' 0 R'for a in annotations)+b'] >>'
    objects[1]=b'<< /Type /Pages /Count 4 /Kids ['+b' '.join(str(a).encode()+b' 0 R'for a in page_ids)+b'] >>'
    out=bytearray(b'%PDF-1.7\n');offsets=[0]
    for i,value in enumerate(objects,1):offsets.append(len(out));out.extend(str(i).encode()+b' 0 obj\n'+value+b'\nendobj\n')
    start=len(out);out.extend(b'xref\n0 '+str(len(objects)+1).encode()+b'\n0000000000 65535 f \n')
    for offset in offsets[1:]:out.extend(f'{offset:010d} 00000 n \n'.encode())
    out.extend(b'trailer\n<< /Size '+str(len(objects)+1).encode()+b' /Root 1 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF\n');generated.write_bytes(out)
    cases=[];paths={str(generated.relative_to(ROOT)):set(range(4))}
    inline=json.loads((ROOT/'build/migration/native-inline-evidence.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    vc={n:json.loads((ROOT/f'build/migration/{n}.json').read_text(encoding='utf-8'))for n in ['native-visual-lines','native-visual-lines-extra']}
    for c in inline['cases']:
        if'source_case'not in c:continue
        r=source['cases'][c['source_case']]['visual_case'];v=vc[r['corpus']]['cases'][r['index']];paths.setdefault(v['pdf'],set()).add(v['page_index'])
    for path,indices in paths.items():
        data=(ROOT/path).read_bytes()
        with PDFDocument(data)as doc:
            for index in sorted(indices):cases.append(dict(pdf=path,pdf_sha256=digest(data),page_index=index,expected=clean([asdict(a)for a in doc.get_page_link_annotations(index)])))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/native_annotations.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_coordinates.py','rapid_doc/_vendor/docvortex_core/foundation/_hyperlink.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_contracts.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},unicode=unicodedata.unidata_version,uris=uris,cases=cases))
    print('Frozen',len(uris),'URI facts and',len(cases),'actual PDF pages /',sum(len(c['expected'])for c in cases),'accepted annotations')
if __name__=='__main__':main()
