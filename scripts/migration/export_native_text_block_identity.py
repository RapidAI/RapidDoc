"""Preserve original list shallow-snapshot identity and equality-based removal exceptions."""
import copy,itertools
from common import ROOT,write_json
from export_native_text_block_edges import block
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-text-block-identity.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.assembly import _merge_list_intro_text_components
    cases=[]
    for ys in itertools.permutations([152.,166.,180.]):
        for mode in ['base','duplicates','protected','mixed']:
            bs=[block('(1) Original item',[30.,y,370.,y+10.])for y in ys]
            if mode=='duplicates':bs.append(copy.deepcopy(bs[-1]))
            if mode=='protected':bs[-1]['_protected_hard_break_before']=True
            if mode=='mixed':bs[-1]['content']='Original introduction'
            c=dict(label=f'list/{ys}/{mode}',request=clean(dict(blocks=bs,page_size=[400.,600.],kind='list')))
            try:c['expected']=clean(_merge_list_intro_text_components(bs))
            except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
            cases.append(c)
    write_json(target,dict(cases=cases));print(len(cases),'list snapshot/equality contracts;',sum('error'in c for c in cases),'original exceptions')
if __name__=='__main__':main()
