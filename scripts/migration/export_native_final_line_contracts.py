"""Unicode boundaries and source-character label geometry for final-line rules."""
import copy,json,sys
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_final_title_rules import reference as rules
from export_native_page_structure import reference as structure
def main():
    target=ROOT/'build/migration/native-final-line-contracts.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();sys.argv.append('--canonical');cases=[]
    f=json.loads((ROOT/'build/migration/native-final-title-rules-typed.json').read_text(encoding='utf-8'))
    def add(label,q,kind='rule'):
        q=copy.deepcopy(q);q['lines']=[project(l)for l in lines(q['lines'])];out=(rules if kind=='rule'else structure)(q);cases.append(dict(label=label,kind=kind,request=q,expected=out))
    front=next(c['request']for c in f['cases']if c['label']=='front/positive')
    for t in ['Original Unıversity','Original Unİversity','Original Inſtitute','Original University']:
        q=copy.deepcopy(front);q['lines'][0]['text']='1 '+t;add('front/'+t,q)
    for marker in ['Received\u0301','Received_','Published\u0301','PublishedA']:
        q=copy.deepcopy(front);q['lines'][-1]['text']=marker;add('boundary/'+marker,q)
    wrapped=next(c['request']for c in f['cases']if c['label']=='wrapped/positive')
    for label in ['A.','١.','12)']:
        q=copy.deepcopy(wrapped);head=q['lines'][-2];head.update(text=label+' Original heading',bbox=[10.,175.,200.,185.],native_title_label_left=None);head['chars']=[dict(char=c,char_idx=i,bbox=[10.+i*4.,175.,14.+i*4.,185.],font=dict(name='Bold',size=10.,flags=0,weight=700.))for i,c in enumerate(head['text'])]
        # Align the suffix to the actual first heading-letter origin, not marker bbox.
        left=10.+(len(label)+1)*4.;tail=q['lines'][-1];tail['wrapped_title_parts'][0]['bbox'][0]=left;tail['wrapped_title_parts'][0]['source_bbox'][0]=left
        add('label-left/'+label,q)
    sf=json.loads((ROOT/'build/migration/native-page-structure.json').read_text(encoding='utf-8'))
    for t in ['INTRODUCTİON','INTRODUCTıON','Referenceſ','ACKNOWLEDGMENTS']:
        q=copy.deepcopy(next(c['request']for c in sf['cases']if c['label']=='explicit/positive'));q['lines'][-1].update(text=t,bbox=[30.,230.,120.,240.]);q['page_size']=[400.,600.];add('explicit-unicode/'+t,q,'structure')
    write_json(target,dict(cases=cases));print(len(cases),'final-line Unicode/label contracts')
if __name__=='__main__':main()
