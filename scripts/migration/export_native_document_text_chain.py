"""Old preparation + auxiliary + complete original document title sequence."""
import copy,json
from common import ROOT,write_json
from export_native_document_auxiliary_chain import reference as auxiliary
from export_native_document_titles import reference as titles

def main():
    target=ROOT/'build/migration/native-document-text-chain.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    f=json.loads((ROOT/'build/migration/native-document-auxiliary-chain.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        q=copy.deepcopy(c['request']);out=dict(label=c['label'],request=q)
        try:
            a=auxiliary(q);t=titles(a);a.update(t);out['expected']=a
        except (ValueError,TypeError,IndexError,AttributeError)as e:out.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(out)
    write_json(target,dict(cases=cases));print(len(cases),'original raw-page-to-document-text chains;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
