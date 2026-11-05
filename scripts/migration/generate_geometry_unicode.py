"""Freeze original Unicode category L/N anchor and run-key classification."""
import sys,unicodedata
from common import ROOT,write_json,digest
def main():
    ranges=[];start=0;previous=0
    for i in range(0x110001):
        f=0 if i==0x110000 else {'L':1,'N':2}.get(unicodedata.category(chr(i))[0],0)
        if f!=previous:
            if previous:ranges.append([start,i-1,previous])
            start=i;previous=f
    p=ROOT/'rust/rapiddoc/src/processing/layout/native_geometry/planner/unicode.json';write_json(p,dict(unicode=unicodedata.unidata_version,ranges=ranges))
    write_json(ROOT/'packaging/geometry-unicode-provenance.json',dict(python=sys.version,unicode=unicodedata.unidata_version,generator='scripts/migration/generate_geometry_unicode.py',sha256=digest(p.read_bytes())))
if __name__=='__main__':main()
