"""Freeze source MTEF font/character maps, not a runtime Python dependency."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.equation import mtef,mtef_v5
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    cp1252={}
    for code in range(256):
        try:cp1252[code]=ord(bytes([code]).decode('cp1252'))
        except UnicodeDecodeError:pass
    data=dict(characters=mtef._CHAR_LATEX,embellishments={i:mtef._apply_embellishments('MTEF_VALUE',(i,)) for i in range(2,38)},symbol=mtef_v5._SYMBOL_FONT_POSITION_TO_UNICODE,
              extra=mtef_v5._MTEXTRA_FONT_POSITION_TO_UNICODE,pua=mtef_v5._MTCODE_PUA_TO_LATEX,cp1252=cp1252)
    data=(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n').encode()
    target=ROOT/'rust/rapiddoc-office/src/equation/mtef_data.json'
    if args.check:assert target.read_bytes()==data
    else:target.write_bytes(data)
    print('MTEF data:',len(data),'bytes')
if __name__=='__main__':main()
