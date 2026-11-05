"""Freeze original canonical em-height value/type preservation."""
from common import ROOT,digest,write_json
from export_table_script_preparation import METRICS
from export_pdf_character_primitives import encode

def main():
    path=ROOT/'build/migration/table-typography-canonical-em.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _fill_native_typography
    cases=[]
    for a in (0,90,180,270):
        for i,em in enumerate([-1,0,1,17,-1.,0.,1.,17.,False,True]):
            v=dict(line=dict(text='A',bbox=[0.,0.,50.,20.],angle=a,chars=[dict(char='A',bbox=[1.,1.,6.,11.])],em_height=em),page_size=[100.,200.])
            l=_LineItem(text='A',bbox=(0.,0.,50.,20.),angle=a,source_index=0,chars=v['line']['chars'],em_height=em);_fill_native_typography(l,(100.,200.))
            cases.append(dict(label=f'em:{a}:{i}',request=v,expected=encode({k:getattr(l,k)for k in METRICS})))
    p='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py';write_json(path,dict(sources={p:digest((ROOT/p).read_bytes())},cases=cases));print('Frozen',len(cases),'canonical em height value/type references')
if __name__=='__main__':main()
