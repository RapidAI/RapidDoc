"""Freeze full native DOCX -> RapidDoc Office middle/export contracts."""
import gzip,json,sys,hashlib
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/migration'))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from rapid_doc.backend.office.model_output_to_middle_json import result_to_middle_json
    from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
    from rapid_doc.version import __version__
    from loguru import logger
    from PIL import Image
    from export_office_path_pixels import normalized_svg
    logger.remove();cases=[];base=ROOT/'tests/fixtures/office_rust_reference_068'
    for fixture in ['docx_body.json.gz','docx_resources.json.gz']:
        for source in json.loads(gzip.decompress((base/fixture).read_bytes()))['cases']:
            case=dict(name=source['name'],fixture=fixture);engine=DocxConverter()
            class Writer:
                def __init__(self):self.data={}
                def write(self,name,data):self.data[name]=data
            writer=Writer()
            try:
                engine.convert(BytesIO(bytes.fromhex(source['package_hex'])));middle=result_to_middle_json(engine.pages,writer)
                def starts(v):
                    if isinstance(v,dict):return ([str(v['start'])]if 'start'in v else[])+sum((starts(x)for x in v.values()),[])
                    if isinstance(v,list):return sum((starts(x)for x in v),[])
                    return []
                case.update(middle=middle,starts=starts(middle),exports={mode:union_make(middle['pdf_info'],mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']})
                assets={}
                for name,raw in writer.data.items():
                    if name.endswith('.svg'):assets[name]=dict(svg=normalized_svg(raw))
                    else:
                        image=Image.open(BytesIO(raw));assets[name]=dict(size=list(image.size),sha256=hashlib.sha256(image.convert('RGBA').tobytes()).hexdigest())
                case['assets']=assets
            except Exception as e:case['error']=type(e).__name__
            cases.append(case)
    path=base/'office_output.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(version=__version__,cases=cases),ensure_ascii=True).encode(),mtime=0));print('Office output references:',len(cases),'documents;',sum('error' in c for c in cases),'errors')
if __name__=='__main__':main()
