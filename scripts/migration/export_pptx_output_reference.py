"""Original PPTX pages plus complete Office middle/assets/four exports."""
import base64,gzip,hashlib,json,sys
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/migration')]
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.pptx_converter import PptxConverter
    from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
    from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
    from rapid_doc.version import __version__
    from export_office_path_pixels import normalized_svg
    from PIL import Image
    from loguru import logger
    logger.remove();base=ROOT/'tests/fixtures/office_rust_reference_068';cases=[]
    sources=json.loads(gzip.decompress((base/'pptx_context.json.gz').read_bytes()))['cases']
    for index,source in enumerate(sources):
        case=dict(name=source['name'],source_index=index)
        class Writer:
            def __init__(self):self.data={}
            def write(self,name,data):self.data[name]=data
        writer=Writer()
        try:
            engine=PptxConverter();engine.convert(BytesIO(base64.b64decode(source['package_b64'])))
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
    (base/'pptx_output.json.gz').write_bytes(gzip.compress(json.dumps(dict(version=__version__,cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('PPTX complete exports:',len(cases),'cases;',sum('error'in c for c in cases),'original errors')
if __name__=='__main__':main()
