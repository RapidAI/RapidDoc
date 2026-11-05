"""Original PPT pages plus complete Office middle/assets/four exports."""
import base64,gzip,hashlib,json,sys
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/migration')]
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt.ppt_converter import PptConverter
    from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
    from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
    from rapid_doc.version import __version__
    from export_office_path_pixels import normalized_svg
    from PIL import Image
    from loguru import logger
    from dataclasses import asdict
    from types import SimpleNamespace
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt.models import PptParagraph, PptTextRun
    logger.remove();base=ROOT/'tests/fixtures/office_rust_reference_068';cases=[]
    sources=[(p.name,p.read_bytes()) for p in [ROOT/'demo/ms_office_docs/pptx_01.ppt',ROOT/'tests/migration/inputs/synthetic.ppt']]
    sys.path.insert(0,str(ROOT/'tests/native_office'))
    from _mtef_test_utils import _build_nested_cfb
    prepared=json.loads(gzip.decompress((base/'ppt_primitives.json.gz').read_bytes()))['prepared']
    seen=set()
    for c in prepared:
        streams=(c['data'],c['current'])
        if streams in seen:continue
        seen.add(streams)
        raw=_build_nested_cfb([dict(name='Root Entry',kind=5,child=1),
            dict(name='PowerPoint Document',kind=2,right=2,data=bytes.fromhex(c['data'])),
            dict(name='Current User',kind=2,data=bytes.fromhex(c['current']))])
        sources.append(('recovery/'+c['name'],raw))
    for name,raw in sources:
        case=dict(name=name,source_hex=raw.hex())
        class Writer:
            def __init__(self):self.data={}
            def write(self,name,data):self.data[name]=data
        writer=Writer()
        try:
            engine=PptConverter();engine.convert(BytesIO(raw))
            case['pages']=engine.pages
            middle=result_to_middle_json(engine.pages,writer)
            def starts(v):
                if isinstance(v,dict):return ([str(v['start'])]if 'start'in v else[])+sum((starts(x)for x in v.values()),[])
                if isinstance(v,list):return sum((starts(x)for x in v),[])
                return []
            case.update(middle=middle,starts=starts(middle),exports={mode:union_make(middle['pdf_info'],mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']})
            assets={}
            for name,raw in writer.data.items():
                if name.endswith('.svg'):assets[name]=dict(svg=normalized_svg(raw))
                else:
                    with Image.open(BytesIO(raw))as im:assets[name]=dict(size=list(im.size),sha256=hashlib.sha256(im.convert('RGBA').tobytes()).hexdigest())
            case['assets']=assets
        except Exception as e:case['error']=type(e).__name__
        cases.append(case)
    (base/'ppt_output.json.gz').write_bytes(gzip.compress(json.dumps(dict(version=__version__,cases=cases),ensure_ascii=True).encode(),mtime=0))
    def paragraph(text='', **kwargs):
        return PptParagraph(runs=(PptTextRun(text),), **kwargs)
    projections=[
        ('breaks',[paragraph('  a\nb\t c  ')]),
        ('links',[PptParagraph(runs=(PptTextRun(' a ',bold=True,hyperlink='https://example.org'),PptTextRun('b',underline=True,hyperlink='https://example.org')))]),
        ('visible-space',[PptParagraph(runs=(PptTextRun('  ',underline=True),PptTextRun('body'),PptTextRun(' ',strike=True)))]),
        ('scripts',[PptParagraph(runs=(PptTextRun('x',baseline=1),PptTextRun(' y',baseline=-1,italic=True)))]),
        ('adjacent-styles',[PptParagraph(runs=(PptTextRun('a',bold=True),PptTextRun('b',bold=True)))]),
        ('nested-types',[paragraph('a',list_kind='ordered',start=3),paragraph('b',list_kind='unordered',depth=1),paragraph('c',list_kind='ordered',depth=1,start=7),paragraph('d',list_kind='ordered')]),
        ('deep',[paragraph('a',list_kind='unordered',depth=12)]),
        ('list-boundaries',[paragraph('a',list_kind='unordered'),paragraph('plain'),paragraph('b',list_kind='unordered')]),
        ('title-after-list',[paragraph('list',list_kind='unordered'),paragraph('heading'),paragraph('body')]),
    ]
    rows=[]
    for name,paragraphs in projections:
        blocks=PptConverter._text_element_blocks(SimpleNamespace(paragraphs=paragraphs),title_candidate=name=='title-after-list')
        for block in blocks:block.pop('_ppt_title_candidate',None)
        rows.append(dict(name=name,paragraphs=[asdict(p) for p in paragraphs],title=name=='title-after-list',blocks=blocks,
                         html=PptConverter._table_cell_content(SimpleNamespace(paragraphs=paragraphs))))
    (base/'ppt_projection.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print('PPT complete exports:',len(cases),'cases;',sum('error'in c for c in cases),'original errors')
if __name__=='__main__':main()
