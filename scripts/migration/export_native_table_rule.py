"""冻结实际旧原生表格捷径守卫/回填与候选仲裁，不运行或替换候选生成器。"""
import copy,json,math,random
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-rule.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import engine
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableCandidate,NativeTableCell
    class ModelRoute(Exception):pass
    class Manager:
        def get_atom_model(self,**kw):raise ModelRoute()
    rng=random.Random(28816);policy=[];saved=old.recover_docvortex_table
    base=dict(poly=[20.,20.,80.,20.,80.,80.,20.,80.],layout_image_list=[{'uuid':'old'}],unknown={'a':1})
    def add(config=None,rotate=None,source=True,ocr=False,formulas=(),checkboxes=(),images=(),result=None,scale=2.):
        table=copy.deepcopy(base);calls=[];method='model'
        request=dict(config=config or{},rotate_label=rotate,has_source_pdf=source,ocr_enable=ocr,formulas=list(formulas),checkboxes=list(checkboxes),images=list(images),table=copy.deepcopy(table),scale=scale)
        def recover(page,bbox):calls.append(list(bbox));return None if result is None else SimpleNamespace(**result)
        page=dict(rotate_label=rotate,ori_image_list=list(images))
        if source:page['_source_pdf_bytes']=b'not opened'
        task=dict(lang=None,useful_list=[0,0,0,0,100,100,100,100],table_res=table,table_img=np.zeros((100,100,3),np.uint8),ocr_enable=ocr,single_page_mfdetrec_res=list(formulas),checkbox_res=list(checkboxes))
        old.recover_docvortex_table=recover
        try:old._process_single_table(task,page,scale,Manager(),config or{},{});method='native'
        except ModelRoute:pass
        policy.append(dict(request=request,result=result,expected=dict(attempt=bool(calls),bbox=calls[0]if calls else None,accepted=method=='native',table=table)))
    try:
        for html in ['', ' \n\x1c ', '<table><tr><td>old</td></tr></table>','<p>not table checked by old gate</p>']:
            for score in [math.nextafter(.9,0),.9,math.nextafter(.9,1),1]:add(result=dict(html=html,confidence=score))
        add()
        for _ in range(512):
            cfg=dict(use_rule_table=rng.choice([True,True,False]),rule_table_engine=rng.choice(['docvortex','docvortex','legacy','off']),force_ocr=rng.choice([False,False,True]),table_formula_enable=False)
            formula=rng.choice([[],[dict(bbox=[30.,30.,40.,40.],latex='x')],[dict(bbox=[0.,0.,40.,40.],latex='x')],[dict(bbox=[80.,80.,100.,100.],latex='x')],[dict(bbox=[81.,81.,100.,100.],latex='x')]])
            image=rng.choice([[],[dict(bbox=[40.,10.,50.,20.])],[dict(bbox=[39.,10.,50.,20.])],[dict(bbox=[0.,0.,10.,10.])]])
            add(config=cfg,rotate=rng.choice([None,'0','90','180','270',90]),source=rng.choice([True,False]),ocr=rng.choice([False,False,True]),formulas=formula if rng.choice([True,False])else[],checkboxes=formula if rng.choice([True,False])else[],images=image,result=dict(html='<table> accepted </table>',confidence=.95))
        # 各守卫边界单独测，不让随机的前置禁用覆盖几何分支。
        for rotate in [None,'0','90','180','270',90]:add(rotate=rotate,result=dict(html='x',confidence=1))
        for f in [[0.,0.,40.,40.],[0.,0.,39.99999999999999,40.],[80.,80.,100.,100.],[81.,81.,100.,100.]]:
            for key in ['formulas','checkboxes']:add(**{key:[dict(bbox=f)]},config={'table_formula_enable':False},result=dict(html='x',confidence=1))
        for b in [[40.,10.,50.,20.],[39.99999999999999,10.,50.,20.],[0.,0.,10.,10.],[0.,0.,10.000000000000002,10.000000000000002]]:add(images=[dict(bbox=b)],result=dict(html='x',confidence=1))
    finally:old.recover_docvortex_table=saved
    sources=list(engine._SOURCE_PRIORITY);selection=[]
    def candidate(source='vector_grid',score=.95,rows=2,cols=2,verified=None):
        cells=tuple(NativeTableCell(i//cols,i%cols,1,1,(0.,0.,1.,1.),'x')for i in range(rows*cols))
        issues=()if verified is None else('evidence='+verified,'ambiguous_separator_ratio=0.0000')
        return NativeTableCandidate(source,rows,cols,cells,score,1.,1.,.8,.8,1.,issues)
    def select(items,remove):
        kept=engine._remove_undercounted_vector_candidates(items)if remove else items
        selected=engine._select_candidate(kept)
        index=lambda c:next(i for i,x in enumerate(items)if x is c)
        accepted=[c for c in kept if engine._passes_verified_threshold(c)]
        selection.append(dict(request=dict(candidates=[asdict(c)for c in items],remove_undercounted=remove),expected=dict(kept=[index(c)for c in kept],selected=None if selected is None else index(selected),physical_topology_conflict=engine._has_line_rect_topology_conflict(accepted))))
    select([],True)
    for source in sources:
        threshold=engine._VERIFIED_SCORE_BY_SOURCE[source]
        for score in [math.nextafter(threshold,0),threshold,math.nextafter(threshold,1),1.]:select([candidate(source,score)],False)
    for score in [math.nextafter(.95,0),.95,math.nextafter(.95,1),1.]:
        for other in [.95,.98,1.]:
            for same in [True,False]:
                for verified in [None,'line_grid','rect_grid']:
                    for remove in [False,True]:select([candidate('vector_grid',score,verified=verified),candidate('sparse_hybrid',other,rows=2 if same else 3)],remove)
    for _ in range(256):
        items=[]
        for i in range(rng.randrange(1,12)):
            items.append(candidate(rng.choice(sources),rng.choice([.8,.94,.95,.98,1.,math.nextafter(.95,0)]),rng.randrange(1,5),rng.randrange(1,5),rng.choice([None,'line_grid','rect_grid'])))
        select(items,rng.choice([False,True]))
    refs=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/docvortex_adapter.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/engine.py']
    write_json(target,dict(policy=policy,selection=selection,sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual old _process_single_table before model initialization with fixed recovery result; actual old candidate threshold/topology/undercount helpers with supplied candidate objects. No native candidate generation or PDF recovery implementation is asserted.'))
    print(f'Frozen {len(policy)} native-route guard/result policies and {len(selection)} old candidate selections')
if __name__=='__main__':main()
