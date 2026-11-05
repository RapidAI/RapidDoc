"""Python max preserves the first equal geometry key, including distinct row shapes."""
from common import ROOT,write_json
from export_native_text_block_edges import block
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-text-block-ties.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import assembly as old
    cases=[]
    groups=[('list',[block('Earlier protected paragraph',[30,170,370,180],_hard_break_before=True),block('Earlier ordinary paragraph',[30,170,370,180]),block('Short tail:',[30,184,100,194]),block('(i) Original list',[30,198,370,208],_hard_break_before=True)]),('list',[block('Earlier ordinary paragraph',[30,170,370,180]),block('Short tail:',[30,184,100,194]),block('Other tail.',[30,184,100,194]),block('(i) Original list',[30,198,370,208],_hard_break_before=True)]),('overlap',[block('Upper ordinary paragraph',[30,160,200,200],rows=[[30,160,200,170],[30,174,200,184],[30,192,200,198],[30,190,200,200]]),block('Lower original fragment',[205,186.5,370,196.5])])]
    for i,(kind,bs)in enumerate(groups):
        for reverse in [False,True]:
            if reverse:
                if kind=='list':bs[0 if i==0 else 1],bs[1 if i==0 else 2]=bs[1 if i==0 else 2],bs[0 if i==0 else 1]
                else:bs[0]['_local_line_bboxes'][-2:]=bs[0]['_local_line_bboxes'][-2:][::-1]
            q=clean(dict(blocks=bs,page_size=[400.,600.],kind=kind));fn=old._merge_list_intro_text_components if kind=='list'else old._merge_overlapping_same_line_text_blocks;out=fn(bs)if kind=='list'else fn(bs,(400.,600.));cases.append(dict(label=f'{kind}/{i}/{reverse}',request=q,expected=clean(out)))
    write_json(target,dict(cases=cases));print(len(cases),'first equal geometry key contracts')
if __name__=='__main__':main()
