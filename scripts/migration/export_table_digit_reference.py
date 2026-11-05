"""冻结CPython Unicode16完整isdigit事实，独立于Rust属性表生成。"""
import unicodedata
from common import ROOT,write_json
def main():
    assert unicodedata.unidata_version=='16.0.0'
    path=ROOT/'build/migration/native-table-digit.json'
    if path.exists():raise FileExistsError(path)
    values=[i for i in range(0x110000)if chr(i).isdigit()]
    write_json(path,dict(unicode=unicodedata.unidata_version,isdigit=values));print(f'Frozen {len(values)} Unicode16 digit codepoints')
if __name__=='__main__':main()
