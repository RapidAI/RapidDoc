"""Freeze printable-empty influence on fraction scale and complete region runs."""
import copy,json
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
 target=ROOT/'build/migration/script-empty-printable.json'
 if target.exists():raise FileExistsError(target)
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _fraction_member_indices,_classify_script_runs
 from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
 f=json.loads((ROOT/'build/migration/script-fractions.json').read_text());base=min((c['request']for c in f['cases']if c['expected']and c['request']['angle']==0),key=lambda v:len(v['chars']))
 fractions=[];runs=[]
 for text in ('',' ','\n'):
  for count in (1,4,40):
   v=copy.deepcopy(base);tight={int(k):q for k,q in v['tight_bboxes'].items()}
   for i in range(count):
    id=1000000+i;b=[150.+i,50.,151.+i,175.];v['chars'].append(dict(char=text,char_idx=id,bbox=b));tight[id]=b
   v['tight_bboxes']=tight
   expected=sorted(_fraction_member_indices(tuple(v['page_size']),v['chars'],tight,v['drawing_lines'],v['angle'],v.get('prepared_rules')))
   fractions.append(dict(label=f'fraction:{text!r}:{count}',request=encode(v),expected=expected))
 f=json.loads((ROOT/'build/migration/script-refinement.json').read_text())
 for n,c in enumerate(f['runs'][:48]):
  v=copy.deepcopy(c['request']);id=1000000+n;b=[200.,50.,201.,175.];v['chars'].append(dict(char='',char_idx=id,bbox=b))
  v.setdefault('tight_bboxes',{})[str(id)]=b;v.setdefault('origins',{})[str(id)]=[200.,175.]
  memberships=v.get('memberships',[None]*(len(v['chars'])-1));v['memberships']=[*memberships,memberships[-1]if memberships else None]
  roles,counts,flags=_classify_script_runs(v['chars'],{int(k):q for k,q in v['tight_bboxes'].items()},{int(k):q for k,q in v['origins'].items()},v['memberships'])
  runs.append(dict(label=f'run:{n}',request=encode(v),expected=dict(roles=roles,body_counts=counts,formula_flags=flags)))
 refs=['inline/scripts.py','_script_geometry.py']
 sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
 write_json(target,dict(sources=sources,fractions=fractions,runs=runs));print('Frozen',len(fractions),'empty/control fraction-scale and',len(runs),'complete run references')
if __name__=='__main__':main()
