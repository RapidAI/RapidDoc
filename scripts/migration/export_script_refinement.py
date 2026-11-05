"""冻结旧token精炼、区域分段和实际line角色整链；不替换内部分类。"""
import copy,json,random,unicodedata
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/script-refinement.json';policy=ROOT/'rust/rapiddoc-native/src/script_geometry/math_identifiers.rs'
    if target.exists()or policy.exists():raise FileExistsError(target)
    assert unicodedata.unidata_version=='16.0.0'
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _refine_math_script_tokens,_classify_script_runs,_script_line_char_roles,_is_math_identifier_char
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._script_geometry import classify_char_script_roles
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    f=json.loads((ROOT/'build/migration/script-geometry.json').read_text(encoding='utf-8'));refine=[];runs=[];lines=[]
    def maps(r):return {int(k):tuple(v)for k,v in r['tight_bboxes'].items()},{int(k):tuple(v)for k,v in r['origins'].items()}
    def add(label,r,raw=None):
        r=copy.deepcopy(r);chars=r['chars'];tight,origins=maps(r)
        base=classify_char_script_roles(chars,tight_bboxes=tight,origins=origins)if raw is None else raw
        for formula in (False,True):
            request=dict(chars=chars,tight_bboxes=tight,origins=origins,roles=base,formula_region=formula,ordinary_native_roles=False)
            expected=_refine_math_script_tokens(chars,base,tight,origins,formula_region=formula)
            refine.append(dict(label=label+f':formula={formula}',request=copy.deepcopy(request),expected=expected))
        for mode in (0,1,2):
            memberships=[None if mode==0 else 0 if mode==1 else (i//5 if i%9<6 else None)for i in range(len(chars))]
            result=_classify_script_runs(chars,tight,origins,memberships)
            runs.append(dict(label=label+f':runs={mode}',request=dict(chars=chars,tight_bboxes=tight,origins=origins,memberships=memberships),expected=dict(roles=result[0],body_counts=result[1],formula_flags=result[2])))
    for i,c in enumerate(f['cases']):
        if i<64 or i%5==0:add(c['label'],c['request'])
    rng=random.Random(629383)
    words=['x1','a_12','A1)','Name12†','[1,2]','［12］','x1 / y2','x1 - y2','pabc-def','∑xy','a×bc','β12','中文x1','x１-２','m^ab','grid','x1.2','a1 + b2']
    for n in range(420):
        text=rng.choice(words);chars=[];tight={};origins={};base=30.
        for i,t in enumerate(text):
            y=base+(0. if i==0 else rng.choice([0.,-3.,-4.,3.]));h=10. if i==0 else rng.choice([10.,6.,8.]);x=float(i*6)
            c=dict(char=t,char_idx=i,bbox=[x,y-h,x+5.,y],font=dict(name='Arial',flags=0,weight=400,size=h))
            chars.append(c);tight[i]=list(c['bbox']);origins[i]=[x,y]
        r=dict(chars=chars,tight_bboxes=tight,origins=origins)
        add(f'token:{n}',r)
        if n%3==0:add(f'given-token:{n}',r,[rng.choice(['body','body','sup','sub'])for c in chars])
        if n%3==0:
            for angle in (0,90,180,270):
                regions=[]if n%2==0 else [[0.,0.,300.,300.]]
                line=_LineItem(text=text,bbox=(0.,0.,300.,300.),angle=angle,source_index=0,chars=copy.deepcopy(chars),inline_math_regions=regions,compact_formula_cluster=n%7==0,restored_inline_cluster=n%11==0)
                fractions={i for i in range(len(chars))if n%13==0 and i%3==0}
                result=_script_line_char_roles(line,(300.,300.),tight,origins,fractions)
                lines.append(dict(label=f'line:{n}:{angle}',request=dict(line=dict(chars=chars,angle=angle,inline_math_regions=regions,compact_formula_cluster=line.compact_formula_cluster,restored_inline_cluster=line.restored_inline_cluster),page_size=[300.,300.],tight_bboxes=tight,origins=origins,fraction_members=sorted(fractions)),expected=dict(chars=encode(result[0]),roles=result[1],body_counts=result[2],formula_flags=result[3])))
    ranges=[];start=last=None
    for code in range(0x110000):
        if 0xd800<=code<=0xdfff or not _is_math_identifier_char(chr(code)):continue
        if last is not None and code==last+1:last=code
        else:
            if last is not None:ranges.append((start,last))
            start=last=code
    if last is not None:ranges.append((start,last))
    policy.write_text('// Generated from actual old _is_math_identifier_char, Unicode16.0.0.\n'+ 'pub const RANGES:&[(u32,u32)]=&[\n'+''.join(f'({a},{b}),\n'for a,b in ranges)+'];\n',encoding='utf-8')
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/scripts.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_script_geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},refine=refine,runs=runs,lines=lines,identifier_ranges=ranges,unicode_version=unicodedata.unidata_version));print('Frozen',len(refine),'refinements,',len(runs),'full runs,',len(lines),'line chains;',len(ranges),'identifier ranges')
if __name__=='__main__':main()
