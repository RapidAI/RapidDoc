"""Complete existing XYCut++ comparison, including sparse/invalid and cross-layout cases."""
import json,random
from common import ROOT,write_json
def main():
    target=ROOT/'build/migration/native-xycut.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native._shared.xycut import sort_entries
    cases=[]
    def add(label,bs):
        for i,b in enumerate(bs):b['test_index']=i
        cases.append(dict(label=label,request=bs,expected=[b['test_index']for b in sort_entries(bs)]))
    rng=random.Random(6194)
    for i in range(180):
        bs=[]
        for j in range(rng.randrange(1,35)):
            x=rng.choice([30.,220.,rng.uniform(0.,350.)]);y=rng.uniform(0,550);bs.append(dict(bbox=[x,y,x+rng.choice([5.,30.,150.,340.]),y+rng.choice([5.,10.,30.,150.])]))
        add('random/'+str(i),bs)
    for name in ['native-visual-annotations','native-final-order-edges']:
        for i,c in enumerate(json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']):
            if i%7==0:add(name+'/'+c['label'],c['request']['blocks'])
    add('invalid',[dict(bbox=b)for b in [None,[1,2,3],[3,4,1,2],[1,1,1,1],[True,False,10,10],['0','0','20','20']]])
    write_json(target,dict(cases=cases));print(len(cases),'complete original XYCut++ cases')
if __name__=='__main__':main()
