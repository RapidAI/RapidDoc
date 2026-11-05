"""Preserve the old interpreter's default 4300-digit int boundary for numbered TOCs."""
import copy,json,sys
from common import ROOT,write_json
from export_native_page_containers import index_base,index_reference
def main():
    target=ROOT/'build/migration/native-index-integer-limits.json'
    if target.exists():raise FileExistsError(target)
    assert sys.get_int_max_str_digits()==4300
    cases=[];errors=[]
    for zero,one in [('0','1'),('٠','١')]:
        for n in [4300,4301]:
            q=index_base('unpaged');q['lines'][1]['text']=zero*(n-1)+one+'. Topic heading';label=f'{ord(zero)}/{n}'
            try:expected=index_reference(q)
            except ValueError:errors.append(dict(label=label,request=copy.deepcopy(q),old_exception='ValueError',error='invalid_input'))
            else:cases.append(dict(label=label,request=copy.deepcopy(q),expected=expected))
    refs=json.loads((ROOT/'build/migration/native-index-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases,errors=errors));print(len(cases),'integer-limit results',len(errors),'old errors')
if __name__=='__main__':main()
