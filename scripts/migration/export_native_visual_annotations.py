"""Full existing visual candidate/parent arbitration and complete binding/region output."""
import copy,json,random
from dataclasses import asdict
from common import ROOT,write_json
from export_native_page_block_rules import typed
from export_native_inline_clusters import clean
from export_native_text_block_edges import block
def reference(q,kind):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import visual_annotations as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.common import _merge_internal_text_block_group
    bs=typed(q['blocks']);p=tuple(q['page_size'])
    for b in bs:
        if '_annotation_band_parent_index'in b:b['_annotation_band_parent']=bs[b['_annotation_band_parent_index']]
    if kind=='bind':
        regions=old._classify_and_bind_visual_annotations(bs,p,merge_text_block_group=_merge_internal_text_block_group);ids={id(b):i for i,b in enumerate(bs)}
        for b in bs:b.pop('_annotation_band_parent',None)
        return clean(dict(blocks=bs,regions=[[ids[id(b)]for b in region]for region in regions]))
    cs=old._collect_annotation_candidates(bs);hs=[old._block_median_line_height(bs[i],p)for i in cs]
    import statistics
    parents=old._build_visual_parents(bs,p,statistics.median(hs)if hs else 8.);old._add_caption_supported_table_panel_parents(bs,p,parents,cs)
    return clean(dict(candidates=list(cs.items()),parents=[asdict(p)for p in parents],relations=[dict(index=i,kind=k,relation=asdict(r)if(r:=old._choose_annotation_relation(i,k,bs,p,parents,set(cs)))is not None else None)for i,k in cs.items()]))
def main():
    target=ROOT/'build/migration/native-visual-annotations.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,bs,p=[400.,600.]):
        for kind in ['relations','bind']:
            q=clean(dict(blocks=copy.deepcopy(bs),page_size=p));c=dict(label=label+'/'+kind,kind=kind,request=q)
            try:c['expected']=reference(q,kind)
            except(ValueError,TypeError,IndexError,KeyError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
            cases.append(c)
    for name in ['native-final-refined-blocks-paths','native-caption-block-rules-canonical']:
        for i,c in enumerate(json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']):
            if 'expected'not in c or i%9:continue
            if isinstance(c['expected'],dict):
                p=c['request']['page'];state=c['expected'];bs=p.get('fixed_blocks',[])+state['formula_blocks']+state['index_blocks']+state['refined_text_blocks'];sz=p.get('page_size',p.get('source',{}).get('page_size',[400.,600.]))
            else:bs=c['expected'];sz=c['request']['page_size']
            add(name+'/'+c['label'],bs,sz)
    image=lambda b:dict(type='image',bbox=list(map(float,b)),content='',angle=0)
    direct=[]
    for t in ['Figure 1. Original caption','Fig. 1 shows original results','图１：原始标题','Table IV. Original caption','Note: Original source','Source(s): Original data','Alpha = original value Source: original dataset','(Smith et al. 2020).','Figure adapted from Original author','Original italic caption with several words']:
        direct.append((t,[image([30,80,370,180]),block(t,[30,184,370,194]),block('Original ordinary body paragraph has many different words and should remain ordinary body text with enough words',[30,215,370,225])]))
    direct.append(('table-footnote',[dict(type='table',bbox=[30.,80.,370.,180.],content='Original table',angle=0),block('Source: Original dataset',[30,184,370,194]),block('Original indented continuation',[40,198,370,208]),block('Original continuation',[40,212,370,222])]))
    direct.append(('components',[image([30,80,190,180]),image([200,80,370,180]),block('Figure 1. Original combined caption',[30,184,370,194])]))
    direct.append(('above',[block('Figure 1. Original graph',[30,64,370,74]),image([30,80,370,180]),dict(type='table',bbox=[30.,20.,370.,60.],content='Original numeric grid',angle=0,_native_numeric_grid=True)]))
    direct.append(('cross-lane',[image([30,80,370,180]),block('Fig. 1 Original left caption',[30,184,180,194],_lane_interval=(30.,180.)),block('Original right caption',[220,184,370,194],_lane_interval=(220.,370.))]))
    direct.append(('bilingual',[image([30,80,370,180]),block('图1 原始标题',[30,184,370,194]),block('Figure 1. Original caption',[30,208,370,218])]))
    direct.append(('definitions',[image([30,80,370,180]),block('ABC = Original definition',[30,184,180,194]),block('Source: Original dataset',[30,198,180,208])]))
    direct.append(('table-panels',[dict(type='table',bbox=[30.,80.,190.,180.],content='Left table',angle=0),dict(type='table',bbox=[200.,80.,370.,180.],content='Right table',angle=0),block('(a) Original left note',[30,184,190,194]),block('(b) Original right note',[200,184,370,194]),block('Table 1. Original combined table',[30,198,370,208])]))
    for n,(label,base)in enumerate(direct):
        for mode in ['base','reverse','different-font','italic','parent-override','blocker','far','rotated','span','no-row','legend']:
            bs=copy.deepcopy(base)
            if mode=='reverse':bs.reverse()
            if mode=='different-font':bs[-1]['_font_signatures']={('Different',64)}
            if mode=='italic':
                for b in bs:
                    if b['type']=='text':b['_font_signatures']={('Body-Italic',64)}
            if mode=='parent-override':
                j=next((i for i,b in enumerate(bs)if b['type']=='image'),None)
                if j is not None:
                    for b in bs:
                        if b['type']=='text':b['_annotation_band_parent_index']=j
            if mode=='blocker':bs.append(block('Original intervening body',[30,181,370,183]))
            if mode=='far':
                for b in bs:
                    if b['type']=='text':b['bbox'][1]+=100;b['bbox'][3]+=100
            if mode=='rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                for b in bs:b.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(b['bbox']),(400.,600.),90))
            if mode=='span':
                for b in bs:b['_lane_is_span']=True
            if mode=='no-row':
                for b in bs:b.pop('_local_line_bboxes',None);b.pop('_line_heights',None)
            if mode=='legend':
                for i,b in enumerate(bs):b['_legend_sort_band']=[200.,30.+i*10.]
            add('direct/'+str(n)+'/'+label+'/'+mode,bs)
    rng=random.Random(4101)
    for i in range(70):
        bs=[image([30,80,370,180])];x=rng.choice([10,30,100,220]);y=rng.choice([50,70,175,184,195,210,250]);bs.append(block(rng.choice(['Figure 1. Original caption','Note: Original dataset','Original body words','Smith 2020.']),[x,y,x+rng.choice([50,150,300]),y+10]));add('random/'+str(i),bs)
    write_json(target,dict(cases=cases));print(len(cases),'full visual relation/binding cases;',sum('error'in c for c in cases),'exceptions;',sum(bool(c.get('expected',{}).get('regions'))for c in cases),'bound regions')
if __name__=='__main__':main()
