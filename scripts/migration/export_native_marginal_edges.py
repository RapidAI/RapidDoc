"""Original page-number/SequenceMatcher contracts and exact marginal thresholds."""
import copy,random
from common import ROOT,write_json
from export_native_marginals import base,reference
from export_native_late_image_text import project
from export_form_runtime import typed
def main():
    target=ROOT/'build/migration/native-marginal-edges.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import auxiliary_text as old
    from loguru import logger
    logger.remove();cases=[];pairs=[];numbers=[]
    for x in ['I','II','ı','İ','ſ','K','1','２','٣','Ⅻ','一','两','〇','零','十','百','MMMM','IIII','IC','ↀ','²']:
        for wrapper in ['{}',' page {} ','- {} -','第{}页','{} of 12','（{}）','{}...']:
            s=wrapper.format(x);value=old._parse_page_number_value(s);numbers.append(dict(text=s,value=None if value is None else str(value)))
        for t in [f'({x}2)',f'{x}.2',f'Chapter {x} ... 12',f'Journal {x} 10']:
            pairs.append(dict(first=t,second=t.replace('2','3'),matches=old._marginal_text_matches(t,t.replace('2','3'))))
    rng=random.Random(333)
    for _ in range(160):
        a=''.join(rng.choice('aabbcc １２中β_İ')for _ in range(rng.randrange(8,100)));b=''.join(rng.choice('aabbcc １２中β_İ')for _ in range(rng.randrange(8,100)));pairs.append(dict(first=a,second=b,matches=old._marginal_text_matches(a,b)))
    for y in [47.9,48.,48.1,89.9,90.,90.1,107.9,108.,108.1,491.9,492.,492.1,539.9,540.,540.1,551.9,552.,552.1]:
        s=base()
        for p in s:p['lines'][0]['bbox']=[30.,y-5.,280.,y+5.];p['lines'][1]['text']='different'+str(y)
        for p in s:p['lines']=[project(l)for l in typed(p).lines]
        cases.append(dict(label=f'center/{y}',request=dict(sources=s),expected=reference(s)))
    for pattern in ['Chapter A .... 12','Chapter A · · · iv','Chapter A\n.... 12','... 12','A.... 12','_(1).....１２','(ı2)','(İ2)','1.2.3']:
        s=base()
        for p in s:p['lines'][0]['text']=pattern;p['lines']=[project(l)for l in typed(p).lines]
        cases.append(dict(label='identity/'+pattern,request=dict(sources=s),expected=reference(s)))
    write_json(target,dict(cases=cases,text=pairs,numbers=numbers));print(len(cases),'boundary documents;',len(pairs),'exact text pairs;',len(numbers),'page-number contracts')
if __name__=='__main__':main()
