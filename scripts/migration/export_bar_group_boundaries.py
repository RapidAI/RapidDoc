"""Additional immutable old threshold and decimal rounding facts for bar groups."""
import copy,math
from common import ROOT,digest,write_json
from export_native_bar_groups import base,reference
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-bar-group-boundaries.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,s):cases.append(dict(label=label,request=clean(dict(source=s,em=10.)),expected=reference(s,10.)))
    for delta in [4.99999,5.,5.00001]:
        s=base()
        for i,p in enumerate(s['path_infos']):p['bbox'][2]=p['bbox'][0]+80.+i*delta/3.
        add('length-spread-'+str(delta),s)
    for width in [59.99999,60.,60.00001]:
        s=base()
        for i,p in enumerate(s['path_infos']):p['bbox'][2]=p['bbox'][0]+width-20.+i*20./3.
        add('core-width-'+str(width),s)
    for height in [29.99999,30.,30.00001]:
        s=base()
        for i,p in enumerate(s['path_infos']):p['bbox'][1]=100.+i*(height-2.)/3.;p['bbox'][3]=p['bbox'][1]+2.
        add('core-height-'+str(height),s)
    for gap in [49.99999,50.,50.00001]:
        s=base();s['lines']=s['lines'][:2]
        for l in s['lines']:l['bbox']=[40.-gap,100.,50.-gap,110.]
        add('numeric-distance-'+str(gap),s)
    for shift in [-.00001,0.,.00001]:
        s=base();s['lines']=s['lines'][:2]
        for l in s['lines']:l['bbox']=[0.,60.-shift,10.,70.-shift]
        add('numeric-hypot-'+str(shift),s)
    for edge in [51.99999,52.,52.00001]:
        s=base();s['lines']=s['lines'][:2];s['lines'][0]['bbox']=[edge-20.,100.,edge,110.];add('overlap-10percent-'+str(edge),s)
    s=base();s['path_infos']*=2;s['lines']=s['lines'][:2];s['lines'][0]['source_index']=s['lines'][1]['source_index'];add('duplicate-bar-and-source-identity',s)
    rounding=[]
    for x in [0.05,0.15,0.25,1.05,1.15,2.55,2.65,-.05,-1.15,1e8+.05]:
        for v in [math.nextafter(x,-math.inf),x,math.nextafter(x,math.inf)]:rounding.append(dict(value=v,expected=round(v,1)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_graphic_geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,rounding=rounding));print('Frozen',len(cases),'threshold cases;',sum(bool(c['expected']['groups'])for c in cases),'positive;',len(rounding),'round(x,1) facts')
if __name__=='__main__':main()
