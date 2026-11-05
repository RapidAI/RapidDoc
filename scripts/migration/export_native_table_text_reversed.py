"""补充冻结旧反向table_bbox的advance几何quirk，不改既有参考。"""
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-text-reversed.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
    chars=[dict(char='a',bbox=[10.,10.,15.,20.],loose_bbox=[9.,8.,20.,22.]),dict(char='b',bbox=[21.,10.,26.,20.],loose_bbox=[20.,8.,30.,22.])]
    cases=[]
    for bbox in [[100.,0.,0.,80.],[0.,80.,100.,0.],[100.,80.,0.,0.]]:
        for angle in [0,90,180,270]:
            value=build_native_table_text(NativeTableInput(tuple(bbox),(100.,80.),angle,tuple(chars)))
            cases.append(dict(request=dict(chars=chars,table_bbox=bbox,angle=angle),expected=None if value is None else asdict(value)))
    source=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/text.py'
    write_json(target,dict(cases=cases,source_sha256=digest(source.read_bytes())))
if __name__=='__main__':main()
