"""冻结实际BatchAnalyze公式阶段的任务顺序与回填；固定响应只验证策略，不代表公式推理。"""
import copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/model-formula-tasks.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
    from loguru import logger
    cases=[];rng=random.Random(1037)
    values=[None,'',' ',r'x^{2}+y^{2}',r'\frac{中文}{αβ}', '\n a \\! b \t',r'\left( x \right)', '$x$']
    with logger.contextualize():
        # 空识别会有旧警告；只关闭此冻结过程的logger输出。
        logger.disable('rapid_doc.backend.pipeline.batch_analyze')
        for n in range(100):
            plans=[];jobs=[];regions=[];outputs=[]
            for page in range(n%5):
                dets=[];tasks=[]
                for i in range(rng.randrange(1,10)):
                    cat=rng.choice([1,3,5,8,13,14]);d=dict(category_id=cat,poly=[0,0,10,0,10,10,0,10],score=.9,unknown={'integer':2**54+1})
                    if rng.randrange(2):d['latex']='prior_'+str(i)
                    dets.append(d)
                    if cat in [8,13,14]:
                        source=f'formula-{page}-{i}'
                        tasks.append(dict(layout_index=i,source=source,useful_list=[0]*8))
                        jobs.append(dict(page=page,layout_index=i,source=source))
                        regions.append(dict(formula_res=d,formula_img=source,lang='en'))
                        outputs.append(values[len(outputs)%len(values)])
                plans.append(dict(model=dict(layout_dets=dets,page_info=dict(page_no=page)),formulas=tasks,ocr=[],tables=[],unknown='preserve'))
            initial=copy.deepcopy(plans);calls=[]
            def predict(images,batch_size):
                calls.append(dict(images=images,batch_size=batch_size));return outputs
            owner=SimpleNamespace(model=SimpleNamespace(formula_model=SimpleNamespace(batch_predict=predict)),formula_base_batch_size=1+n%4)
            BatchAnalyze._run_formula_recognition(owner,regions)
            cases.append(dict(request=dict(plans=initial,results=outputs),expected=plans,jobs=jobs,calls=calls,batch_size=owner.formula_base_batch_size))
        logger.enable('rapid_doc.backend.pipeline.batch_analyze')
    ref='rapid_doc/backend/pipeline/batch_analyze.py'
    write_json(target,dict(schema=1,scope='fixed responses; actual legacy formula stage only; no formula network inference',reference=dict(path=ref,sha256=digest((ROOT/ref).read_bytes())),cases=cases))
    print(f'Frozen {len(cases)} actual legacy formula-stage cases')

if __name__=='__main__':main()
