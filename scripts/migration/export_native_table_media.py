"""Freeze original media geometry/masks/unique-cell insertion, including declined ownership."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    path=ROOT/'build/migration/native-table-media.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PageSource,PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_media import nested_table_media,mask_table_media,attach_table_media
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableText,NativeTableCell,NativeTableResult,NativeTableRule,NativeTableRectangle
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_native_table_html
    cases=[]
    for n in range(96):
        a=(0,90,180,270)[n%4];size=(200.,200.);table=(0.,0.,200.,200.)
        boxes=[[10.,10.,20.,20.],[49.8,10.,50.2,20.],[45.,10.,55.,20.],[100.,100.,150.,150.]][(n//4)%4]
        def rot(b):return tuple(_rotate_bbox_from_upright(tuple(b),size,a))
        media=[rot(boxes)];images=[*media,*media]if n%3 else[rot([0.,0.,200.,200.]),*media]
        paths=[PDFPathInfo(rot(boxes),6 if n%5 else 4,True,False,1,0)]
        if n%7==0:paths.append(PDFPathInfo(rot(boxes),4,False,True,1,1))
        source=_PageSource(size,[],[],[],image_bboxes=images,form_bboxes=media,path_infos=paths)
        chars=[dict(char_idx=0,char='X<&>',bbox=rot(boxes),tight_bbox=None),dict(char_idx=1,char='outside',bbox=rot([160.,160.,180.,170.]))]
        rules=(NativeTableRule(rot(boxes),.2,'horizontal'),NativeTableRule(rot([0.,0.,200.,.1]),.1,'horizontal'))
        ti=NativeTableInput(table,size,a,tuple(chars),rules,(NativeTableRectangle(rot(boxes),6,True,True,1),))
        cells=tuple(NativeTableCell(0,i,1,1,(i*50.,0.,(i+1)*50.,200.),str(i))for i in range(4))
        html=serialize_native_table_html(1,cells)
        if n%11==0:html=html.replace('</td>','</td>',1).replace('<td>','<TD>',1)
        result=NativeTableResult(html,1,4,cells,NativeTableText((),(),1.,1.),'text_grid'if n%13==0 else'vector_grid',1.)
        selected=nested_table_media(source,table);masked=mask_table_media(ti,selected);attached=attach_table_media(html,result,ti,selected)
        cases.append(dict(label=str(n),source=dict(form_bboxes=list(source.form_bboxes),image_bboxes=images,path_infos=[asdict(p)for p in paths]),table_bbox=table,table_input=asdict(ti),result=asdict(result),content=html,selected=selected,masked=asdict(masked),attached=None if attached is None else dict(content=attached[0],images=attached[1])))
    prefix='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'
    write_json(path,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in ['table_media.py','geometry.py','_table_recovery/geometry.py']},cases=clean(cases)))
    print('Frozen',len(cases),'old media facts;',sum(c['attached']is not None for c in cases),'attached;',sum(c['attached']is None for c in cases),'declined')
if __name__=='__main__':main()
