"""冻结未mock的旧原始字符→完整候选引擎/恢复/诊断，不覆写参考。"""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-engine.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import engine as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page
    cases=[];real=[]
    def add(v,label):
        ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v['chars']),tuple(NativeTableRule(**(r|dict(bbox=tuple(r['bbox']))))for r in v['drawing_lines']),tuple(NativeTableRectangle(**(r|dict(bbox=tuple(r['bbox']))))for r in v['rectangles']))
        evaluation=old._evaluate_native_pdf_table(ti);result=old.recover_native_pdf_table(ti);diagnostics=old.diagnose_native_pdf_table(ti)
        cases.append(dict(label=label,request=dict(table_input=copy.deepcopy(v)),expected=dict(evaluation=asdict(evaluation),result=None if result is None else asdict(result),diagnostics=diagnostics)))
    def raw(r):
        v=copy.deepcopy(r['table_input'])
        if not v.get('chars'):
            x0,y0,x1,y1=v['table_bbox'];w=x1-x0;h=y1-y0;angle=int(v['angle'])%360
            def undo(b):
                a,b0,c,d=b
                if angle==90:return [x0+w-d,y0+a,x0+w-b0,y0+c]
                if angle==180:return [x0+w-c,y0+h-d,x0+w-a,y0+h-b0]
                if angle==270:return [x0+b0,y0+h-c,x0+d,y0+h-a]
                return [x0+a,y0+b0,x0+c,y0+d]
            v['chars']=[dict(char_idx=g['source_index'],char=g['text'],bbox=undo(g['bbox']))for g in r['text']['glyphs']]
        return v
    for name,step in [('native-table-vector-pipeline',3),('native-table-rule-band',4),('native-table-sparse-hybrid',8),('native-table-multiline',8),('native-table-multiline-extra',1),('native-table-banded-numeric',10),('native-table-banded-numeric-extra',1),('native-table-banded-numeric-ties',1)]:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases'][::step]:
            r=c['request'];v=raw(r)
            if len(v['drawing_lines'])>1000 or len(v['rectangles'])>1000:continue
            # 真正运行build_native_table_text，不把分支测试的预构建text灌入引擎。
            add(v,name+':'+str(c.get('label','synthetic-glyphs')))
    base=copy.deepcopy(cases[0]['request']['table_input']);base['chars']=[]
    for bbox in [[0.,0.,0.,100.],[-.1,0.,100.,100.],[0.,0.,2001.,100.]]:v=copy.deepcopy(base);v['table_bbox']=bbox;v['page_size']=[2000.,2000.];add(v,'input-geometry')
    add(base,'no-native-text')
    v=copy.deepcopy(base);v.update(table_bbox=[0.,0.,100.,100.],page_size=[200.,200.],drawing_lines=[dict(bbox=[1.,1.,99.,1.1],width=.1,orientation='horizontal')]*5001);add(v,'primitive-limit')
    # 独立旧DocVortex进程提取实际PDF字符/线/矩形；全部字符字段以普通JSON冻结。
    pdfs=[('build/migration/model-document/five-pages.pdf',2,(18.,38.,222.,162.)),('tests/migration/inputs/pdf_native.pdf',0,(20.,200.,180.,320.)),('build/migration/aligned-numeric-amounts.pdf',0,(10.,20.,440.,160.)),('build/migration/aligned-numeric-status.pdf',0,(10.,20.,440.,160.))]
    for name,page,bbox in pdfs:
        pth=ROOT/name
        with PDFDocument(pth.read_bytes())as doc:
            p=prepare_table_page(doc[page]);chars=[]
            for c in p.geometry.chars:
                q={}
                for k in ('char_idx','char','bbox','loose_bbox','tight_bbox','font','writing_angle','origin'):
                    if k in c:
                        value=c[k];q[k]=list(value.bbox)if hasattr(value,'bbox')else value
                chars.append(q)
            for angle in (0,90,180,270):
                v=dict(table_bbox=list(bbox),page_size=list(p.page_size),angle=angle,chars=chars,drawing_lines=[asdict(r)for r in p.drawing_lines],rectangles=[asdict(r)for r in p.rectangles]);i=len(cases);add(v,'actual-pdf');real.append(dict(pdf=name,pdf_sha256=digest(pth.read_bytes()),page_index=page,angle=angle,case_index=i))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/';modules=['engine.py','text.py','vector.py','candidate.py','aligned_numeric.py','rule_band.py','sparse_hybrid.py','text_grid.py','sparse_multiline.py','grouped_rule_band.py','banded_numeric.py'];write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in modules},cases=cases,real=real));print(f'Frozen {len(cases)} complete engine raw inputs; {sum(c["expected"]["result"]is not None for c in cases)} recoveries, {len(real)} actual PDF/angle raw inputs')
if __name__=='__main__':main()
