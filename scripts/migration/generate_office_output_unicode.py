"""Freeze CPython HTML entities and Unicode punctuation for native Office export."""
import html,json,unicodedata
from html.entities import html5
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    destination=ROOT/'rust/rapiddoc-office/src/document_output'
    destination.joinpath('html_entities.json').write_text(json.dumps(dict(entities=html5,invalid_map=html._invalid_charrefs,invalid=sorted(html._invalid_codepoints)),ensure_ascii=True,separators=(',',':'))+'\n',encoding='utf-8')
    ranges=[];start=last=None
    for n in range(0x110000):
        if unicodedata.category(chr(n))[0]in 'PS':
            if last is not None and n==last+1:last=n
            else:
                if start is not None:ranges.append([start,last])
                start=last=n
    if start is not None:ranges.append([start,last])
    destination.joinpath('punctuation.json').write_text(json.dumps(ranges,separators=(',',':'))+'\n',encoding='utf-8')
    print('Python HTML entities and Unicode punctuation:',unicodedata.unidata_version,len(ranges),'ranges')
if __name__=='__main__':main()
