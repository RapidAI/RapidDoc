"""Existing set membership, numeric identities and default source-index semantics."""
import copy
from common import ROOT,write_json
from export_native_spacing import reference
def main():
    target=ROOT/'build/migration/native-spacing-contracts.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for a,b in [(0,1),(False,True),(0.,1.),(None,1),('0',1),(2,3.),(-2,-1),(9007199254740992,9007199254740993),(9007199254740992,9007199254740992.),('missing',-1)]:
        for boundary in [b,float(b),str(b)]:
            cs=[dict(char='a',char_idx=a),dict(char='b',char_idx=b)]
            if a=='missing':cs[0].pop('char_idx')
            q=dict(lines=[dict(chars=cs,bbox=[0.,0.,100.,10.],source_index=0)],space_before=[boundary]);c=dict(label=repr((a,b,boundary)),kind='prepare',request=q)
            try:c['expected']=reference(q,'prepare')
            except(ValueError,TypeError,KeyError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
            cases.append(c)
    write_json(target,dict(cases=cases));print(len(cases),'spacing identity contracts;',sum('error'in c for c in cases),'exceptions')
if __name__=='__main__':main()
