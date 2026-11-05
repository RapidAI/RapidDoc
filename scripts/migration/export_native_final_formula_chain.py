"""Raw source/document/reference preparation followed by exact final-formula prefix."""
import copy,json
from common import ROOT,write_json
from export_native_reference_chain import reference as prepare_references
from export_native_final_formula import reference as formulas

def reference(q):
    out=prepare_references(q)
    out['formula_pages']=[formulas(dict(page=p,page_index=i))for i,p in enumerate(out['pages'])]
    return out

def main():
    target=ROOT/'build/migration/native-final-formula-chain.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    f=json.loads((ROOT/'build/migration/native-reference-chain.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        q=copy.deepcopy(c['request']);out=dict(label=c['label'],request=q)
        try:out['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:out.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(out)
    write_json(target,dict(cases=cases));print(len(cases),'original raw-source/document/reference/final-formula chains;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
