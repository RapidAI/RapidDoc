"""Original XLSX normalization/retry and package boundary references."""
import gzip,json,sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from export_xlsx_resources_reference import normalize

def main():
    from openpyxl import Workbook
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xlsx.package_normalizer import normalize_xlsx_package
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xlsx.xlsx_converter import XlsxConverter
    from loguru import logger
    logger.remove();cases=[]
    def tree(n):return [n.tag,dict(sorted(n.attrib.items())),n.text,[tree(c) for c in n]]
    def edit(data,updates):
        out=BytesIO()
        with ZipFile(BytesIO(data))as z,ZipFile(out,'w',ZIP_DEFLATED)as dest:
            for n in z.namelist():
                value=updates.get(n);raw=z.read(n)
                dest.writestr(n,value(raw.decode()).encode()if callable(value)else value if value is not None else raw)
            for n,v in updates.items():
                if n not in z.namelist():dest.writestr(n,v)
        return out.getvalue()
    def capture(name,data):
        case=dict(name=name,package_hex=data.hex())
        try:
            repaired=normalize_xlsx_package(data);changed={}
            with ZipFile(BytesIO(data))as a,ZipFile(BytesIO(repaired))as b:
                for n in b.namelist():
                    if a.read(n)!=b.read(n):changed[n]=tree(ET.fromstring(b.read(n)))
            case['normalized']=changed
            engine=XlsxConverter();engine.convert(BytesIO(data));case['pages']=normalize(engine.pages)
        except Exception as e:case['error']=type(e).__name__
        cases.append(case)
        return case
    wb=Workbook();wb.active['B2']='header';wb.active['D3']=10
    out=BytesIO();wb.save(out);base=out.getvalue()
    for dimension in ['B2:D3','A:A','C5','$B$1:$F$22','bad','','ZZZ1:ZZZ7']:
        for ref in ['2:3','$2:$3','3:2','0:3','foo','A1:D3','A:B','A1:B','A0:B0']:
            capture('filter '+repr((dimension,ref)),edit(base,{'xl/worksheets/sheet1.xml':lambda s,d=dimension,r=ref:s.replace('ref="B2:D3"','ref="'+d+'"').replace('</worksheet>','<autoFilter ref="'+r+'"/></worksheet>')}))
    for fill in ['<fill/>','<fill></fill>','<fill>text</fill>','<fill><gradientFill/></fill>']:
        capture('fill '+fill,edit(base,{'xl/styles.xml':lambda s,f=fill:s.replace('<fill><patternFill/></fill>',f)}))
    # A manifest-discovered SST, independent of its relationship presence.
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    for attrs in ['val=""','val=" "','val="double"','val="bad"','s:val=""','val="" s:val="double"','val="double" s:val=""']:
        sst=f'<sst xmlns="{ns}" xmlns:s="{ns}"><si><r><rPr><u {attrs}/></rPr><t>rich</t></r></si></sst>'
        capture('shared underline '+attrs,edit(base,{
            'xl/sharedStrings.xml':sst,
            '[Content_Types].xml':lambda s:s.replace('</Types>',f'<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>'),
            'xl/worksheets/sheet1.xml':lambda s:s.replace('t="n"><v>10</v>','t="s"><v>0</v>')}))
    for filename in ['xlsx_resources.json.gz','xlsx_cells.json.gz']:
        fixtures=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068'/filename).read_bytes()))['cases']
        if filename=='xlsx_resources.json.gz':
            for source in [c for c in fixtures if c['name']in ['images PNG table','chart BarChart','OMML table','OLE native']]:
                data=bytes.fromhex(source['package_hex'])
                recovered=capture('empty title '+source['name'],edit(data,{'xl/workbook.xml':lambda s:s.replace('name="Data"','name=""')}))
                if source['name']=='OLE native' and recovered.get('error')=='KeyError':
                    # Native OLE does not need openpyxl's stripped loading copy;
                    # preserve the formula instead of reproducing that crash.
                    recovered['native_recovery_pages']=source['queries'][0]['pages']
                capture('bad optional image '+source['name'],edit(data,{'xl/media/image1.png':b'not an image'}))
        else:
            for source in fixtures:
                if source.get('error'):capture('preserved '+source['name'],bytes.fromhex(source['package_hex']))
    dest=ROOT/'tests/fixtures/office_rust_reference_068/xlsx_compat.json.gz'
    dest.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('XLSX compat:',len(cases),'cases;',sum('error'in c for c in cases),'original errors')
if __name__=='__main__':main()
