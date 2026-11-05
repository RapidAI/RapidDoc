"""Original output/tight normalization, preserving invalid-row all-or-empty behavior."""
import copy,json
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-final-output.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from loguru import logger
    logger.remove();cases=[]
    def add(label,bs,p,kind):
        q=clean(dict(blocks=bs,page_size=p,kind=kind));c=dict(label=label,request=q)
        try:
            bs=typed(q['blocks'])
            if kind=='tight':old._apply_post_aggregation_tight_bboxes(bs,tuple(p));out=bs
            elif kind=='lines':out=[old._normalize_output_line_items(b,tuple(p))for b in bs]
            else:out=[old._normalize_output_block(b,tuple(p))for b in bs]
            c['expected']=clean(out)
        except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for name in ['native-final-refined-blocks-paths','native-page-block-rules','native-caption-block-rules-canonical']:
        for i,c in enumerate(json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']):
            if 'expected'not in c or i%5:continue
            bs=c['expected'].get('refined_text_blocks',[])if isinstance(c['expected'],dict)else c['expected'];p=c['request'].get('page_size',c['request'].get('page',{}).get('page_size',[400.,600.]))
            for kind in ['tight','lines','normalize']:add(name+'/'+c['label']+'/'+kind,bs,p,kind)
    base=dict(type='text',content='Original paragraph',bbox=[30.,180.,370.,190.],angle=0,_local_line_bboxes=[[30.,180.,370.,190.]])
    modes=[('tiny',dict(bbox=[.0001,.0001,.0002,.0002])),('edge',dict(bbox=[399.99,599.99,400.,600.])),('inverted',dict(bbox=[370.,190.,30.,180.])),('outside',dict(bbox=[500.,700.,600.,800.])),('invalid',dict(bbox=[1,2,3])),('controls',dict(content='a\u200bb\r\nc\u0000d\u00ade')),('empty',dict(content='\x1c')),('equation',dict(type='equation',content='Original equation')),('image',dict(type='image',content='',angle=90)),('unknown',dict(type='other')),('angle',dict(angle=-90)),('angle-string',dict(angle='90')),('line-inverted',dict(_local_line_bboxes=[[370.,190.,30.,180.]])),('line-invalid',dict(_local_line_bboxes=[[1,2,3]])),('line-bool',dict(_local_line_bboxes=[[True,False,10,10]])),('regions',dict(_inline_math_regions=[[40.,180.,50.,190.],[1,2,3],[370,190,30,180]])),('parent',dict(_native_annotation_parent_bbox=[-40,-20,500,700])),('table-images',dict(type='table',_table_images=[dict(bbox=[-10.,-10.,500.,700.],name='original')])),('boundary',dict(_protected_hard_break_before=True)),('tight',dict(_tight_output_bbox=[31.,181.,369.,189.])),('repaired',dict(_local_output_line_bboxes=[[31.,181.,369.,189.]],_output_bbox_repaired=True)),('bad-repair',dict(_tight_output_bbox=[31.,181.,369.,189.],_local_output_line_bboxes=[[1,2,3]],_output_bbox_repaired=True))]
    for name,changes in modes:
        b=dict(copy.deepcopy(base),**changes)
        for a in [0,90,180,270]:
            b=copy.deepcopy(b);b['angle']=changes.get('angle',a)
            for kind in ['tight','lines','normalize']:add('direct/'+name+'/'+str(a)+'/'+kind,[b],[400.,600.],kind)
    write_json(target,dict(cases=cases));print(len(cases),'full original tight/line/output cases;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
