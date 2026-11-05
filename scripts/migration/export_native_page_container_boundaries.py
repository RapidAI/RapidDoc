"""Independent rotation, continuation, section and Form source-map boundaries."""
import copy,json
from common import ROOT,write_json
from export_native_page_containers import index_base,index_reference,form_base,forms_reference,page_reference

def main():
    target=ROOT/'build/migration/native-page-container-boundaries.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(kind,label,q,fn):cases.append(dict(kind=kind,label=label,request=copy.deepcopy(q),expected=fn(q)))
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    for a in [0,90,180,270]:
        for kind in ['paged','leaders','unpaged']:
            q=index_base(kind)
            for l in q['lines']:l['bbox']=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),a));l['angle']=a
            add('index',f'rotation/{a}/{kind}',q,index_reference)
    for mode in ['continuation','section-continuation','numeric-continuation','wrong-font','bool-font','gap-low','gap-high','repeated-chapter','mixed-chapter','heading-ignored-for-leaders','tiny-number','newline-numbered','cjk-unpaged','roman-only','leader-glyph-height']:
        q=index_base('unpaged'if mode in ['tiny-number','newline-numbered','cjk-unpaged']else'leaders'if mode in ['heading-ignored-for-leaders','leader-glyph-height']else'paged');ls=q['lines']
        if mode in ['continuation','section-continuation','numeric-continuation','wrong-font','bool-font','gap-low','gap-high']:
            extra=[]
            for l in ls[1:]:
                l['bbox'][2]=200.;l['text']='Topic heading';extra.append(dict(l,text=str(l['source_index']+1),bbox=[450.,l['bbox'][1],460.,l['bbox'][3]],source_index=l['source_index']+1000))
                cont=dict(l,text='continued label',bbox=[40.,l['bbox'][3]+3.,190.,l['bbox'][3]+10.],effective_height=10.,source_index=l['source_index']+2000,visual_row_id=l['source_index']+2000)
                if mode=='section-continuation':cont['text']='Chapter next'
                if mode=='numeric-continuation':cont['text']='1.)More'
                if mode=='wrong-font':cont['font_signature']=['Other',0]
                if mode=='bool-font':cont['font_signature']=['Body',False]
                if mode=='gap-low':cont['bbox'][1]=l['bbox'][3]-3.;cont['bbox'][3]=cont['bbox'][1]+7.
                if mode=='gap-high':cont['bbox'][1]=l['bbox'][3]+8.;cont['bbox'][3]=cont['bbox'][1]+7.
                extra.append(cont)
            ls+=extra
        if mode in ['repeated-chapter','mixed-chapter']:
            # Chapter rows are internal non-page rows; first repeated section can be restored.
            for l in ls[1:]:l['bbox'][1]+=20.;l['bbox'][3]+=20.
            for j,y in enumerate([100.,150.,190.]):ls.append(dict(ls[1],text=f'Chapter {j+1}: Heading'if mode=='repeated-chapter'else'Chapter I1: Heading',bbox=[40.,y,180.,y+10.],source_index=3000+j,visual_row_id=3000+j))
        if mode=='heading-ignored-for-leaders':ls.pop(0);q.update(require_heading=True,require_explicit_heading=True)
        if mode=='tiny-number':ls.append(dict(ls[1],text='999',bbox=[40.,125.,45.,128.],effective_height=3.,source_index=999,visual_row_id=999))
        if mode=='newline-numbered':
            for l in ls[1:]:l['text']=f"{l['source_index']+1}. Topic\nlabel"
        if mode=='cjk-unpaged':
            for l in ls[1:]:l['text']=f"{l['source_index']+1}. 原生解析主题"
        if mode=='roman-only':
            for l in ls[1:]:l['text']='iv'
        if mode=='leader-glyph-height':
            for l in ls[1:]:l['effective_height']=2.;l['chars']=[dict(char='A',bbox=[40.,l['bbox'][1],45.,l['bbox'][3]])]
        add('index',mode,q,index_reference)
    for mode in ['unordered-ids','duplicate-map','empty-mapped-clipped','alias-existing','thin-evidence','all-tight-required','source-missing','no-visible','form-info-float','path-outside-members']:
        s=form_base();b=s['form_bboxes'][0];s['form_member_sources']=[dict(bbox=b,source_indices=[1,0])];s['form_path_sources']=[dict(bbox=b,source_indices=[1,0])]
        if mode=='duplicate-map':s['form_member_sources']=[dict(bbox=b,source_indices=[]),dict(bbox=b,source_indices=[0,1])]
        if mode=='empty-mapped-clipped':s['form_bboxes']=[[-20.,-10.,360.,360.]];s['form_member_sources']=[dict(bbox=s['form_bboxes'][0],source_indices=[])]
        if mode=='alias-existing':s['form_member_sources'].append(dict(bbox=[60.,100.,340.,340.],source_indices=[99]))
        if mode=='thin-evidence':
            for p in s['path_infos']:p['bbox']=[100.,150.,110.,160.]
            for d in s['drawing_lines']:d['bbox']=[100.,150.,110.,150.]
        if mode=='all-tight-required':
            for l in s['lines']:l['bbox']=[-100.,l['bbox'][1],600.,l['bbox'][3]];l['chars'].append(dict(char='x',char_idx=0))
        if mode=='source-missing':
            for l in s['lines']:
                for c in l['chars']:c.pop('char_idx')
        if mode=='no-visible':
            for l in s['lines']:
                for c in l['chars']:c['char']='\u0085'
        if mode=='form-info-float':s['form_bboxes']=[[0,0,500,600]];s['form_infos']=[dict(bbox=[0.,0.,500.,600.])]
        if mode=='path-outside-members':s['form_path_sources'][0]['source_indices']=[]
        add('forms',mode,dict(source=s),forms_reference)
    refs=json.loads((ROOT/'build/migration/native-index-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases,errors=[]));print(len(cases),'boundary facts')

if __name__=='__main__':main()
