"""Freeze pinned Apache POI DOC data and Python public contracts for regression.

Source inputs and original Apache LICENSE/NOTICE remain under the specified corpus
directory. The resulting fixture includes their exact bytes, not implementation code.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from PIL import Image
from rapid_doc import RapidDoc
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
from export_office_path_pixels import normalized_svg
from rapid_doc.version import __version__

def starts(value):
    if isinstance(value,dict):return ([str(value['start'])] if 'start' in value else [])+sum((starts(v) for v in value.values()),[])
    if isinstance(value,list):return sum((starts(v) for v in value),[])
    return []

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--corpus',type=Path,default=ROOT/'build/doc-external-corpus');args=parser.parse_args()
    from loguru import logger
    logger.remove()
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    cases=[]
    for c in manifest['cases']:
        data=(args.corpus/c['name']).read_bytes();assert hashlib.sha256(data).hexdigest()==c['sha256']
        result=RapidDoc()(data)
        assets={}
        for name,raw in result.images.items():
            if name.endswith('.svg'):assets[name]={'svg':normalized_svg(raw)}
            else:
                image=Image.open(io.BytesIO(raw)).convert('RGBA')
                assets[name]={'size':list(image.size),'sha256':hashlib.sha256(image.tobytes()).hexdigest()}
        cases.append(dict(name=c['name'],source_hex=data.hex(),middle=result.middle_json,starts=starts(result.middle_json),assets=assets,
            exports={mode:union_make(result.middle_json['pdf_info'],mode,'images') for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']}))
    target=ROOT/'tests/fixtures/office_rust_reference_068'
    (target/'doc_external.json.gz').write_bytes(gzip.compress(json.dumps({'version':__version__,'source_manifest':manifest,'cases':cases},ensure_ascii=True).encode(),mtime=0))
    for name in ['LICENSE','NOTICE']:(target/f'APACHE-POI-DOC-{name}.txt').write_bytes((args.corpus/name).read_bytes())
    print('Independent POI DOC documents:',len(cases))

if __name__=='__main__':main()
