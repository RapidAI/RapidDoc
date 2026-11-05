"""实际旧分数线检测→单行完整角色处理，验证来源ID和抑制后的计数时序。"""
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/script-fraction-pipeline.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _fraction_member_indices,_script_line_char_roles
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    size=(300.,300.);cases=[]
    for shift in (0.,.1,2.,8.,20.):
        for angle in (0,90,180,270):
            chars=[];tight={};origins={}
            for idx,text,x,y,h in [(100,'A',0.,70.,10.),(101,'B',10.,70.,10.),(104,'x',20.,61.,6.),(107,'y',20.,79.,6.)]:
                x+=shift;b=_rotate_bbox_from_upright((x,y-h,x+6.,y),size,angle)
                o=(x,y)if angle==0 else (size[0]-y,x)if angle==90 else(size[0]-x,size[1]-y)if angle==180 else(y,size[1]-x)
                chars.append(dict(char=text,char_idx=idx,bbox=list(b),font=dict(name='Arial',flags=0,weight=400,size=h)));tight[idx]=b;origins[idx]=o
            rule=_rotate_bbox_from_upright((19.+shift,69.75,27.+shift,70.25),size,angle)
            members=_fraction_member_indices(size,chars,tight,[rule],angle)
            line=_LineItem(text='ABxy',bbox=(0.,0.,300.,300.),source_index=0,chars=chars,angle=angle,inline_math_regions=[(0.,0.,300.,300.)])
            def output(m):
                c,r,n,f=_script_line_char_roles(line,size,tight,origins,m);return dict(chars=encode(c),roles=r,body_counts=n,formula_flags=f)
            cases.append(dict(label=f'{shift}:{angle}',fraction=dict(chars=chars,tight_bboxes=tight,drawing_lines=[list(rule)],page_size=list(size),angle=angle),line=dict(line=dict(chars=chars,angle=angle,inline_math_regions=[[0.,0.,300.,300.]]),page_size=list(size),tight_bboxes=tight,origins=origins),expected_members=sorted(members),before=output(set()),expected=output(members)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/scripts.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_script_geometry.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'complete fraction-to-line chains;',sum(c['before']['roles']!=c['expected']['roles']for c in cases),'actual suppression changes')
if __name__=='__main__':main()
