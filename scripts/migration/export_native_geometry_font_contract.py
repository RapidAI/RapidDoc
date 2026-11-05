"""Full producer font conversion behavior, including original conversion failures."""
import copy
from common import ROOT,write_json
from export_native_geometry_plans import page,reference
from export_native_geometry import lines
from export_native_late_image_text import project
def main():
    target=ROOT/'build/migration/native-geometry-font-contract.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    for field,value in [('flags','１_０００'),('flags','+０'),('flags','1.0'),('flags','\x1c1'),('flags',1.9),('size','bad'),('size','nan'),('size','inf'),('size','\x1cinf'),('weight','nan'),('weight','-nan'),('weight','inf'),('weight','bad'),('weight','４５０'),('name',''),('font',[]),('font',[1]),('font',False),('font',0)]:
        p=page(cell=14.)
        for l in p['source']['lines']:
            for c in l['chars']:
                if field=='font':c['font']=copy.deepcopy(value)
                else:c['font'][field]=value
        p['source']['lines']=[project(l)for l in lines(p['source']['lines'])];q=dict(pages=[p]);item=dict(label=f'{field}/{value!r}',request=q)
        try:item['expected']=reference(q)
        except (ValueError,OverflowError,AttributeError,TypeError)as e:item['error']=type(e).__name__
        cases.append(item)
    write_json(target,dict(cases=cases));print(len(cases),'full-producer font contracts;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
