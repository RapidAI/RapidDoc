"""Reproduce the Adobe glyph-name and Unicode category tables used for NUL recovery."""
import hashlib
import json
from pathlib import Path
import unicodedata
from importlib.metadata import version
from fontTools.agl import LEGACY_AGL2UV

ROOT=Path(__file__).resolve().parents[2]
DESTINATION=ROOT/'rust/rapiddoc-pdf/src/unicode_fallback'

def main():
    DESTINATION.mkdir(parents=True,exist_ok=True)
    (DESTINATION/'glyphs.json').write_text(json.dumps(LEGACY_AGL2UV,sort_keys=True,separators=(',',':')),encoding='utf8')
    ranges=[]
    for code in range(0x110000):
        if unicodedata.category(chr(code)).startswith('C'):
            if ranges and ranges[-1][1]+1==code:ranges[-1][1]=code
            else:ranges.append([code,code])
    (DESTINATION/'category_c.json').write_text(json.dumps(ranges,separators=(',',':')),encoding='utf8')
    metadata={'source':'fontTools.agl.LEGACY_AGL2UV','fonttools':version('fonttools'),'unicode_version':unicodedata.unidata_version,'source_license':'../../../../licenses/fonttools-glyph-data','files':{name:hashlib.sha256((DESTINATION/name).read_bytes()).hexdigest() for name in ('glyphs.json','category_c.json')}}
    (DESTINATION/'SOURCES.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
