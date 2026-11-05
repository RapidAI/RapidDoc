"""Add positive blank-answer fixtures; retain the earlier negative header-gap corpus."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_form_tables import base,reference
def main():
    target=ROOT/'build/migration/native-blank-answer-positive.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def fixture():
        q=base('blank_answer');q['source']['lines'][0]['chars']=[dict(char='a',bbox=[x,105.,x+5.,115.])for x in [50.,57.,64.,160.,167.,174.]];return q
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    q=fixture();assert reference(q)['candidates'];add('positive',q)
    for mode in ['not-bold','no-signature','missing-chars','wide-key','terminal','extra-gap','unicode-caption']:
        q=fixture();s=q['source']
        if mode=='not-bold':s['lines'][0]['font_signature']=['Arial',0]
        if mode=='no-signature':s['lines'][0]['font_signature']=None
        if mode=='missing-chars':s['lines'][0]['chars']=[]
        if mode=='wide-key':s['lines'][1]['bbox'][2]=150.
        if mode=='terminal':s['lines'][1]['paragraph_terminal']=True
        if mode=='extra-gap':s['lines'][2]['bbox'][1]+=10.
        if mode=='unicode-caption':s['lines'][-1]['text']='Table ١. Data'
        add(mode,q)
    rng=random.Random(271031)
    for i in range(40):
        q=fixture();rng.shuffle(q['source']['lines']);rng.shuffle(q['source']['drawing_lines']);l=rng.choice(q['source']['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    p='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py';write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())},cases=cases));print('Frozen',len(cases),'blank-answer facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
