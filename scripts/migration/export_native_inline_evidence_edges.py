"""Additional immutable old references for compact multi-character list markers."""
import copy
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-inline-evidence-edges.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.detection import detect_pdf_text_style_lines,detect_pdf_text_link_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    cases=[]
    for text in ['•• ','•\u00ad•','••\u200b','\u200b••','• \t•']:
        chars=[dict(char=text,char_idx=0,bbox=[10.,20.,20.,30.],font=dict(name='Bold',flags=0,weight=700)),dict(char='body',char_idx=1,bbox=[26.,20.,56.,30.],font=dict(name='Helvetica',flags=0,weight=400))]
        line=dict(text=text+'body',source_index=0,angle=0,bbox=[10.,20.,56.,30.],chars=chars)
        for mixed in [False,True]:
            lines=[line]
            if mixed:
                other=copy.deepcopy(line);other.update(source_index=1,bbox=[10.,50.,56.,60.]);other['chars']=[dict(char='Bold',char_idx=2,bbox=[10.,50.,56.,60.],font=dict(name='Bold',flags=0,weight=700))];lines.append(other)
            objects=[_LineItem(**copy.deepcopy(l))for l in lines]
            expected=clean(dict(style_lines=[asdict(s)for s in detect_pdf_text_style_lines(objects,[])],link_lines=[asdict(s)for s in detect_pdf_text_link_lines(objects,[])]))
            cases.append(dict(label=repr(text)+str(mixed),request=dict(lines=lines,drawing_lines=[],annotations=[]),expected=expected))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/common.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'compact-marker cases')
if __name__=='__main__':main()
