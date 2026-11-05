"""冻结Unicode16 isdigit超出decimal的128码点，运行时不依赖宿主Unicode。"""
import unicodedata
from common import ROOT
def main():
    assert unicodedata.unidata_version=='16.0.0'
    values=[i for i in range(0x110000)if chr(i).isdigit()and not chr(i).isdecimal()]
    path=ROOT/'rust/rapiddoc/src/processing/table/table_native_vector/pipeline/digit.rs'
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('//! Generated from CPython Unicode16 isdigit minus isdecimal.\npub const EXTRA: &[u32] = &[\n'+''.join(f'    {i},\n'for i in values)+'];\n',encoding='utf-8')
    print(f'Frozen {len(values)} extra digit codepoints')
if __name__=='__main__':main()
