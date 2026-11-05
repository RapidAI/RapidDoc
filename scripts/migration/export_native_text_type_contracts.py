"""Preserve first Python numeric/boolean set representatives in block metadata."""
from common import ROOT,write_json
from export_native_title_layout import line
from export_native_text_blocks import reference
def main():
    target=ROOT/'build/migration/native-text-type-contracts.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for ids in [[True],[False],[True,1],[1,True],[False,0],[0,False],[False,True,0,1]]:
        for note in [False,True]:
            ls=[line('Original ordinary paragraph words',[30.,180.+14*i,370.,190.+14*i],i,visual_row_id=v,semantic_type='page_footnote'if note else None)for i,v in enumerate(ids)]
            q=dict(lines=ls,page_size=[400.,600.],table_bboxes=[],drawing_lines=[],visual_bboxes=[],page_footnote_groups=[list(range(len(ls)))]if note else[],page_index=0)
            cases.append(dict(label=f'visual/{ids}/{note}',request=q,expected=reference(q)))
    for flag in [True,False]:
        ls=[dict(line('Original ordinary paragraph words',[30.,180.+14*i,370.,190.+14*i],i),font_signature=['Body',v])for i,v in enumerate([flag,int(flag)])]
        q=dict(lines=ls,page_size=[400.,600.],table_bboxes=[],drawing_lines=[],visual_bboxes=[],page_footnote_groups=[],page_index=0);cases.append(dict(label=f'font/{flag}',request=q,expected=reference(q)))
    write_json(target,dict(cases=cases));print(len(cases),'first numeric/boolean set representative contracts')
if __name__=='__main__':main()
