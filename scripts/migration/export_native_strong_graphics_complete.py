"""Correct typed path projection defaults; retain the first frozen exporter attempt for traceability."""
import json,copy
from common import ROOT,write_json
from export_native_strong_graphics import reference
def main():
    target=ROOT/'build/migration/native-strong-graphics-complete.json'
    if target.exists():raise FileExistsError(target)
    old=json.loads((ROOT/'build/migration/native-strong-graphics.json').read_text(encoding='utf-8'));cases=[];errors=[]
    for c in old['cases']+old['errors']:
        q=copy.deepcopy(c['request'])
        try:out=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=c['label'],request=q,error='invalid_input',old_exception=type(e).__name__));continue
        cases.append(dict(label=c['label'],request=q,expected=out))
    write_json(target,dict(sources=old['sources'],scope='Actual old complete strong graphic function with correctly reconstructed PDFPathInfo default rectangle_bboxes; first frozen exporter attempt preserved but its constructor errors are not detector evidence.',cases=cases,errors=errors));print('Complete strong',len(cases),'errors',len(errors),'positives',sum(bool(c['expected']['graphic_bboxes'])for c in cases),'mutations',sum(c['request']['source']!=c['expected']['source']for c in cases))
if __name__=='__main__':main()
