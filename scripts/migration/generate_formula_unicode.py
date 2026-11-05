"""Freeze Python's Unicode properties used by native formula text features."""
import sys,unicodedata
from common import ROOT,write_json,digest

def main():
    math=[];digits=[]
    for n in range(0x110000):
        c=chr(n)
        if unicodedata.category(c)=='Sm' or 'VULGAR FRACTION'in unicodedata.name(c,''):math.append(n)
        if c.isdecimal():digits.append(n)
    def ranges(values):
        result=[]
        for n in values:
            if result and result[-1][1]+1==n:result[-1][1]=n
            else:result.append([n,n])
        return result
    path=ROOT/'rust/rapiddoc/src/processing/formula/native_vector_formulas/unicode.json'
    write_json(path,dict(unicode=unicodedata.unidata_version,math=ranges(math),decimal=ranges(digits)))
    write_json(ROOT/'packaging/formula-unicode-provenance.json',dict(python=sys.version,unicode=unicodedata.unidata_version,generator='scripts/migration/generate_formula_unicode.py',source_sha256=digest(path.read_bytes())))
if __name__=='__main__':main()
