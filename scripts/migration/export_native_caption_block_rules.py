"""Full existing caption/note/wrap/list rules and exact post-assembly sequence."""
import copy,json,sys
from types import SimpleNamespace
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_final_text_lines import typed_lines
from export_native_inline_clusters import clean
from export_native_title_layout import line
from export_native_text_block_edges import block
from export_native_late_image_text import project
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import assembly as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import annotations as a
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import merge_overlapping_member_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_lists import split_local_list_blocks
    bs=typed(q['blocks']);p=tuple(q['page_size']);images=[tuple(b)for b in q['image_bboxes']];grids=[tuple(b)for b in q['numeric_grid_bboxes']];paths=[SimpleNamespace(**dict(b,bbox=tuple(b['bbox']),rectangle_bboxes=[tuple(r)for r in b.get('rectangle_bboxes',[])]))for b in q['bullet_paths']];kind=q['kind']
    if kind=='captions':return clean(a._merge_image_caption_text_blocks(bs,images,grids))
    if kind=='wrap':return clean(old._restore_caption_wrap_text(bs,images,p))
    if kind=='notes':return clean(a._split_labeled_visual_note_blocks(bs,images))
    if kind=='lists':return clean(split_local_list_blocks(bs,paths))
    assert kind=='sequence'
    bs=merge_overlapping_member_blocks(bs,p);bs=a._merge_multiline_title_blocks(bs);bs=a._merge_front_matter_column_blocks(bs,p,page_index=q['page_index']);bs=a._merge_image_caption_text_blocks(bs,images,grids);bs=a._merge_fragmented_header_blocks(bs);bs=old._restore_caption_wrap_text(bs,images,p);bs=a._merge_repeated_compact_title_continuations(bs,p);bs=a._split_labeled_visual_note_blocks(bs,images);bs=split_local_list_blocks(bs,paths);return clean(bs)
def main():
    target=ROOT/('build/migration/native-caption-block-rules-canonical.json'if '--canonical'in sys.argv else'build/migration/native-caption-block-rules.json')
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,bs,kind,**kw):
        q=dict(blocks=clean(bs),page_size=[400.,600.],page_index=0,image_bboxes=[],numeric_grid_bboxes=[],bullet_paths=[],kind=kind);q.update(kw);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for i,c in enumerate(json.loads((ROOT/'build/migration/native-text-blocks.json').read_text(encoding='utf-8'))['cases']):
        if 'expected'not in c or i%11:continue
        for kind in ['captions','wrap','notes','lists','sequence']:add(c['label']+'/'+kind,c['expected'],kind,page_size=c['request']['page_size'],page_index=c['request'].get('page_index',0),image_bboxes=c['request']['visual_bboxes'])
    def with_lines(texts,boxes,**kw):
        ls=[project(l)for l in typed_lines([line(t,b,i)for i,(t,b)in enumerate(zip(texts,boxes))])];bounds=[min(b[0]for b in boxes),min(b[1]for b in boxes),max(b[2]for b in boxes),max(b[3]for b in boxes)];return block(' '.join(texts),bounds,rows=boxes,_text_lines=ls,**kw)
    directs=[]
    directs.append(('captions',[block('Fig. 1 Original figure',[60,180,340,190]),block('Original first caption tail',[60,194,340,204]),block('Original second caption tail.',[60,208,340,218])],dict(image_bboxes=[[50.,60.,350.,178.]])))
    directs.append(('captions',[block('Fig. 1 Original figure',[60,180,340,190]),block('(Original credit)',[120,194,280,204])],dict(image_bboxes=[[50.,60.,350.,178.]])))
    directs.append(('captions',[block('Fig. 1 Original graph',[60,210,340,220]),block('Original chart caption',[60,224,340,234])],dict(image_bboxes=[[10.,10.,40.,40.]],numeric_grid_bboxes=[[50.,60.,350.,178.]])))
    directs.append(('captions',[block('Fig. 1 Original group caption',[40,180,360,190]),block('Original continued caption',[40,194,360,204])],dict(image_bboxes=[[30.,80.,190.,178.],[200.,80.,370.,178.]])))
    directs.append(('captions',[block('Fig. 1 Original caption',[60,180,340,190]),with_lines(['Original short title tail'],[[60.,194.,340.,204.]],type='paragraph_title')],dict(image_bboxes=[[50.,60.,350.,178.]])))
    directs.append(('notes',[with_lines(['Note: Original note','Source: Original source','Original wider body'],[[30.,180.,370.,190.],[30.,194.,370.,204.],[30.,208.,370.,220.]])],dict(image_bboxes=[[30.,60.,370.,178.]])))
    directs[-1][1][0]['_text_lines'][-1].update(effective_height=12.,em_height=12.)
    directs.append(('lists',[with_lines(['a. Original item','b. Original item','c. Original item'],[[30.,180.,370.,190.],[30.,194.,370.,204.],[30.,208.,370.,218.]])],{}))
    directs.append(('lists',[with_lines(['Original item','Original item','Original item'],[[30.,180.,370.,190.],[30.,194.,370.,204.],[30.,208.,370.,218.]])],dict(bullet_paths=[dict(bbox=[24.,y+3.,27.,y+6.],fill_visible=True,stroke_visible=False,rectangle_bboxes=[],segment_count=12)for y in[180.,194.,208.]])))
    directs.append(('wrap',[block('Fig. 1 Original caption',[200,180,370,194]),with_lines(['Original preceding narrow body','Original narrow words continue','original full-width continuation','Original full-width following'],[[30.,166.,190.,176.],[30.,180.,190.,190.],[30.,194.,370.,204.],[30.,208.,370.,218.]])],dict(image_bboxes=[[200.,60.,370.,178.]])))
    for i,(kind,bs,kw)in enumerate(directs):
        for mode in ['base','reverse','protected','font','heading','split-gap','same-row','mirror','image-away','rule','source-collision']:
            v=copy.deepcopy(bs);opts=copy.deepcopy(kw)
            if mode=='reverse':v.reverse()
            if mode=='protected':v[-1]['_protected_hard_break_before']=True
            if mode=='font':v[-1]['_font_signatures']={('Other',64)}
            if mode=='heading':v[-1]['type']='paragraph_title'
            if mode=='split-gap':
                for b in v:b['_line_heights']=[h*2 for h in b['_line_heights']]
            if mode=='same-row':
                for b in v:
                    for l in b.get('_text_lines',[]):l.update(visual_row_id=7,split_from_row=True)
            if mode=='source-collision':
                for b in v:
                    for l in b.get('_text_lines',[]):l['source_index']=7
            if mode=='rule':v[-1]['_rule_break_before']=True
            if mode=='image-away':opts['image_bboxes']=[[10.,10.,30.,30.]]
            if mode=='mirror':
                def mirror(b):return[400.-b[2],b[1],400.-b[0],b[3]]
                for b in v:
                    b['bbox']=mirror(b['bbox']);b['_local_line_bboxes']=[mirror(x)for x in b['_local_line_bboxes']];b['_local_output_line_bboxes']=[mirror(x)for x in b['_local_output_line_bboxes']]
                    for l in b.get('_text_lines',[]):
                        for key in ['bbox','source_bbox','ink_bbox']:
                            if l.get(key)is not None:l[key]=mirror(l[key])
                opts['image_bboxes']=[mirror(b)for b in opts.get('image_bboxes',[])];opts['numeric_grid_bboxes']=[mirror(b)for b in opts.get('numeric_grid_bboxes',[])]
            add(f'direct/{kind}/{i}/{mode}',v,kind,**opts);add(f'sequence/{kind}/{i}/{mode}',v,'sequence',**opts)
    write_json(target,dict(cases=cases));print(len(cases),'caption/wrap/note/list and complete post-assembly sequences;',sum('error'in c for c in cases),'exceptions;',sum(c['expected']!=c['request']['blocks']for c in cases if 'expected'in c),'positive changes')
if __name__=='__main__':main()
