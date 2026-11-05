"""Original shared source mutations while post-geometry labels/fractions are consumed."""
import copy
from common import ROOT,write_json
from export_native_geometry_pages import reference
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_table_page_policies import empty

def main():
    target=ROOT/'build/migration/native-geometry-aliases.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,ls,graphic=[],rules=[],plan=None):
        ls=[project(l)for l in lines(ls)];s=dict(empty(),page_size=[400.,600.],lines=ls,drawing_lines=rules)
        p=dict(source=s,remaining_lines=copy.deepcopy(ls),strong_graphic_bboxes=[],table_bboxes=[],claimed_line_indices=[],graphic_blocks=graphic)
        q=dict(prepared=p,plan=plan,page_index=0,style_lines=[],link_lines=[]);cases.append(dict(label=label,request=copy.deepcopy(q),expected=reference(q)))
    def label(text,x,i):return dict(text=text,bbox=[x,200.,x+len(text)*5.,210.],angle=0,source_index=i,visual_row_id=3,run_index=i,effective_height=10.,chars=[dict(char=c,char_idx=100*i+j,bbox=[x+5*j,200.,x+5*j+5.,210.],font=dict(name='Body',size=10.,flags=0))for j,c in enumerate(text)])
    for mode in ['label-cache','label-cache-plan','split-source-id','kept-source-id']:
        ls=[label('native label',30.,0),label('other label',150.,1)];graphics=[dict(type='image',bbox=[20.,180.,300.,230.],angle=0,content='original')]
        plan={}if mode=='label-cache-plan'else None
        if mode=='split-source-id':ls[0]['source_index']=7;ls[1]['source_index']=8
        if mode=='kept-source-id':ls[0]['semantic_type']='caption'
        add(mode,ls,graphics,plan=plan)
    from export_native_leading_fractions import base
    q=base();q['lines'][1]['bbox'][3]=218.;q['lines'][1]['chars']=[dict(char='β',char_idx=100,bbox=[50.,200.,58.,218.],font=dict(name='Body',size=8.,flags=0))];add('fraction-source-cache',q['lines'],rules=q['drawing_lines'])
    write_json(target,dict(cases=cases));print(len(cases),'source identity/cache continuations')
if __name__=='__main__':main()
