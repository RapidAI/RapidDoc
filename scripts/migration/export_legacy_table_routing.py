"""实际旧RapidTableModel路由与原HTML选优；独立冻结，不覆写参考。"""
import copy,json,sys
from types import SimpleNamespace
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
from common import ROOT,digest,write_json
def main():
    forced='--compare-real'in sys.argv
    target=ROOT/('build/migration/legacy-table-routing-compare.json'if forced else'build/migration/legacy-table-routing.json')
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import RapidTable,RapidTableInput,ModelType,EngineType
    from rapid_doc.model.table.rapid_table_self.table_cls.main import TableCls
    from rapid_doc.model.table.utils import select_best_table_model,count_table_cells_physical
    prior=json.loads((ROOT/'build/migration/table-quality.json').read_text(encoding='utf-8'))
    selection=[]
    def add(texts,wired,wireless):
        def metrics(s):
            count=count_table_cells_physical(s);blank=sum(not c.text.strip()for c in BeautifulSoup(s,'html.parser').find_all(['td','th']))
            return dict(physical_cells=count,blank_cells=blank,non_blank_cells=count-blank,ocr_hits=sum(t in s for t in texts))
        selection.append(dict(texts=texts,wired_html=wired,wireless_html=wireless,html=select_best_table_model([[],texts],wired,wireless),wired=metrics(wired),wireless=metrics(wireless)))
    import html
    for c in prior['cases']:add([html.escape(t).replace('&#x27;','&#39;')for t in c['ocr_texts']],c['wired_html'],c['wireless_html'])
    cells=['<b>A</b>','&nbsp;','<!--comment-->','<img src="uuid">','<script>A</script>','<template>A</template>','<![CDATA[A]]>','&#x27;','<td>nested</td>','&notit;']
    for value in cells:
        for other in ['', 'A', '<b>A</b>', '&nbsp;']:
            add(['A','&amp;','',value],'<table><thead><tr><td>'+value+'</td></tr></thead></table>','<table><tr><td>'+other+'</td></tr></table>')
    rows=[[[[2,2],[30,2],[30,20],[2,20]]],['中 文 &amp;'],[1.]]
    policy=[];image=np.full((40,80,3),255,np.uint8)
    for label in ['wired','wireless']:
        for score in [.899999999999,.9,1.]:
            for compare in [False,True]:
                model=RapidTableModel.__new__(RapidTableModel);model.model_type=ModelType.UNET_SLANET_PLUS;model.use_compare_table=compare;model.cls_conf_threshold=.9
                model.table_cls=lambda *a,**k:([label],[score],0.)
                calls=[]
                def candidate(name):
                    def run(*a,**kw):calls.append(name);return SimpleNamespace(pred_htmls=['<table><tr><td>'+('中 文 &amp;'if name=='wired'else'other')+'</td></tr></table>'])
                    return run
                model.wired_table_model=candidate('wired');model.wireless_table_model=candidate('wireless')
                final=model.predict(image,copy.deepcopy(rows),skip_table_orientation=True)
                policy.append(dict(label=label,score=score,compare=compare,ocr_result=rows,calls=calls,html=final))
    cls=json.loads((ROOT/'build/migration/table-classification.json').read_text(encoding='utf-8'))
    unet=json.loads((ROOT/'build/migration/legacy-unet.json').read_text(encoding='utf-8'))
    slanet=json.loads((ROOT/'build/migration/legacy-table-match.json').read_text(encoding='utf-8'))['real'][0]
    model=RapidTableModel.__new__(RapidTableModel);model.model_type=ModelType.UNET_SLANET_PLUS;model.use_compare_table=forced;model.cls_conf_threshold=.9
    cfg=dict(engine_type=EngineType.ONNXRUNTIME,use_ocr=False,engine_cfg={'intra_op_num_threads':1})
    model.wired_table_model=RapidTable(RapidTableInput(model_type=ModelType.UNET,model_dir_or_path=str(ROOT/unet['model']),**cfg))
    model.wireless_table_model=RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS,model_dir_or_path=str(ROOT/slanet['model']),**cfg))
    model.table_cls=TableCls(RapidTableInput(model_type=ModelType.PADDLE_Q_CLS,model_dir_or_path={'paddle_cls':str(ROOT/cls['model']),'q_cls':str(ROOT/cls['qmodel'])},**cfg))
    real=[]
    for c in unet['real']:
        with Image.open(ROOT/c['source'])as im:image=np.asarray(im.convert('RGB'))
        pred=model.predict(image,copy.deepcopy(c['ocr_result']),skip_table_orientation=True)
        real.append(dict(source=c['source'],source_sha256=c['source_sha256'],ocr_result=c['ocr_result'],html=pred))
    refs=['rapid_doc/model/table/rapid_table.py','rapid_doc/model/table/utils.py','rapid_doc/model/table/rapid_table_self/table_cls/main.py']
    write_json(target,dict(selection=selection,policy=policy,real=real,classifiers=cls,unet_model=unet['model'],unet_sha256=unet['sha256'],slanet_model=slanet['model'],slanet_sha256=slanet['sha256'],
        sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual old raw HTML selection and RapidTableModel call order/threshold policies, five real Paddle+Qanything/UNET/SLANet candidates with provided OCR. Default blank-cell rec disabled; does not prove native rule-table shortcut or complete production parser parity.'))
    print(f'Frozen {len(selection)} old HTML selections, {len(policy)} routing policies and {len(real)} actual CPU dual-model routes')
if __name__=='__main__':main()
