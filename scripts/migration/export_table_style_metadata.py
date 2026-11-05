"""固定旧字体元数据数值/字符串转换，避免用宽松float解析替代int。"""
from dataclasses import asdict
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/table-style-metadata.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _render_styled_cell
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableCell,NativeTableGlyph
    glyph=NativeTableGlyph(0,0,'X',(0.,0.,6.,10.),0);cell=NativeTableCell(0,0,1,1,(0.,0.,6.,10.),'X',(0,));cases=[]
    for name in ['Arial','BOLDXX+Arial','demİ']:
        for flag in [0,64,262144,-1,True,None,'64','64.9','６４','64.0','0e0',(1<<60)|64,float((1<<60)+262144)]:
            for weight in [None,False,400,600,599.9,'600.0','６００','nan','inf',10**15]:
                chars={0:dict(char='X',char_idx=0,bbox=[0.,0.,6.,10.],font=dict(name=name,flags=flag,weight=weight,size=10.))}
                r=dict(cell=asdict(cell),glyphs=[asdict(glyph)],median_height=10.,roles={},chars_by_source=chars,tight_bboxes={},origins={},preserve_font_styles=True,possible_space_indices=None)
                expected=_render_styled_cell(cell,[glyph],10.,{},chars_by_source=chars,preserve_font_styles=True)
                cases.append(dict(label=f'{name}:{flag!r}:{weight!r}',request=r,expected=expected))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py';write_json(target,dict(sources={source:digest((ROOT/source).read_bytes())},cases=cases));print('Frozen',len(cases),'font metadata conversions')
if __name__=='__main__':main()
