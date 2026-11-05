"""Freeze original rich text grouping, stripping, Unicode slicing and formulas."""
import gzip,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.formatting_types import Formatting
    from rapid_doc._vendor.docvortex_core.content.spans import strip_span_dicts,slice_span_dicts
    from loguru import logger
    logger.remove();rng=random.Random(310313);engine=DocxConverter();cases=[]
    formats=[None,{},dict(bold=True),dict(italic=True,underline=True),dict(underline=True,underline_style='words'),dict(emphasis=True,strikethrough=True),dict(script='super'),dict(script='sub')]
    texts=['',' ',' \t ','hello','中文😀','\r\n','\r','x\nY','²¹','\u001c',' <b> & ']
    links=[None,'#local','https://example.com/a','javascript:bad','../relative','https://example.com/a']
    for _ in range(450):
        elements=[];request=[]
        for _ in range(rng.randrange(1,9)):
            text=rng.choice(texts);fmt=rng.choice(formats);link=rng.choice(links);format_obj=Formatting(**fmt) if fmt is not None else None
            elements.append((text,format_obj,link));request.append(dict(text=text,format=format_obj.model_dump(mode='json') if format_obj else None,hyperlink=link))
        spans=engine._build_text_from_elements(elements);a,b=sorted([rng.randrange(25),rng.randrange(25)])
        raw=''.join(e[0] for e in elements);cut=rng.randrange(len(raw)+1);tokens=[('text',raw[:cut]),('equation',' x^2 '),('text',raw[cut:])]
        formula=engine._build_text_with_equations_and_hyperlinks(elements,raw,tokens)
        cases.append(dict(elements=request,spans=spans,stripped=strip_span_dicts(spans),slice=[a,b],sliced=slice_span_dicts(spans,a,b),raw=raw,tokens=tokens,formula=formula))
    path=ROOT/'tests/fixtures/office_rust_reference_068/docx_spans.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0));print('DOCX span references:',len(cases))
if __name__=='__main__':main()
