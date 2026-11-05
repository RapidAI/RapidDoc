"""Freeze original enclosing-fill detector and int/float area decisions without overwriting."""
import copy,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def reference(paths,page,excluded=()):
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_filled_grid_table_candidates
    ps=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox']),rectangle_bboxes=tuple(tuple(b)for b in p.get('rectangle_bboxes',[]))))for p in paths]
    return clean(dict(candidates=[asdict(c)for c in _detect_filled_grid_table_candidates(ps,tuple(page),[tuple(b)for b in excluded])]))
def path(b):return dict(bbox=b,form_depth=0,fill_visible=True,stroke_visible=False,segment_count=5,source_index=0,rectangle_bboxes=[])
def base(x=40.,y=100.,w=200.,h=120.,rows=4,cols=2):
    return [path([x,y,x+w,y+h])]+[path([x+c*w/cols,y+r*h/rows,x+(c+1)*w/cols,y+(r+1)*h/rows])for r in range(rows)for c in range(cols)]
def main():
    target=ROOT/'build/migration/native-filled-grid.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,paths=None,page=(400.,600.),excluded=()):
        ps=copy.deepcopy(paths if paths is not None else base());q=clean(dict(path_infos=ps,page_size=page,excluded_bboxes=excluded));cases.append(dict(label=label,request=q,expected=reference(ps,page,excluded)))
    add('base');add('empty',[])
    for rows in [1,2,3,4,5,8]:add('rows-'+str(rows),base(rows=rows))
    for cols in [1,2,3,4]:add('columns-'+str(cols),base(cols=cols))
    for page in [(0.,600.),(400.,0.),(-400.,600.),(400.,-600.),(-400.,-600.)]:add('page-'+str(page),page=page)
    for key,values in [('form_depth',[0,1]),('fill_visible',[False,True]),('segment_count',[4,5,6])]:
        for value in values:
            ps=base()
            for p in ps:p[key]=value
            add(key+str(value),ps)
    for count in [0,1,2]:
        ps=base()
        for p in ps:p['rectangle_bboxes']=[[0.,0.,1.,1.]]*count
        add('rectangle-count-ignored-'+str(count),ps)
    for height in [19.99999,20.,20.00001]:add('minimum-area-'+str(height),base(w=60.,h=height))
    for height in [199.99999,200.,200.00001]:add('maximum-area-'+str(height),base(w=300.,h=height))
    for width in [47.99999,48.,48.00001]:add('minimum-width-'+str(width),base(w=width,h=40.))
    for height in [17.99999,18.,18.00001]:add('minimum-height-'+str(height),base(w=100.,h=height))
    for left in [139.99999,140.,140.00001]:add('excluded-'+str(left),excluded=[[left,100.,left+200.,220.]])
    for dy in [22.49999,22.5,22.50001]:
        ps=base()
        for r in range(4):
            for c in range(2):ps[1+r*2+c]['bbox'][1]=106.+r*(dy+6.);ps[1+r*2+c]['bbox'][3]=106.+r*(dy+6.)+dy
        add('vertical-coverage-'+str(dy),ps)
    for gap in [5.99999,6.,6.00001]:
        ps=base()
        for r in range(4):
            for c in range(2):ps[1+r*2+c]['bbox'][1]=105.+r*(23.+gap);ps[1+r*2+c]['bbox'][3]=128.+r*(23.+gap)
        add('vertical-gap-'+str(gap),ps)
    for gap in [1.99999,2.,2.00001]:
        ps=[path([40.,100.,240.,220.])]+[path([40.+c*(20.+gap),100.+r*30.,60.+c*(20.+gap),130.+r*30.])for r in range(4)for c in range(9)];add('horizontal-gap-'+str(gap),ps)
    ps=base();ps+=copy.deepcopy(ps);add('duplicate-paths',ps)
    ps=base();ps+=copy.deepcopy(ps[1:])
    for p in ps[9:]:p['bbox'][3]=(p['bbox'][1]+p['bbox'][3])/2
    add('half-row-shading',ps)
    for change in [1.07999,1.08,1.08001]:
        ps=base();extra=copy.deepcopy(ps[1:])
        for p in extra:p['bbox'][3]=p['bbox'][1]+(p['bbox'][3]-p['bbox'][1])/change
        add('maximal-cell-'+str(change),ps+extra)
    for shift in [.99999,1.,1.00001]:
        ps=base()
        for p in ps[1::2]:p['bbox'][0]+=shift
        add('cell-gap-'+str(shift),ps)
    ps=base();ps+=base(x=270.,y=300.,w=80.,h=100.);add('two-tables',ps)
    for scale in [.01,1.,1000000.]:
        ps=base()
        for p in ps:p['bbox']=[x*scale for x in p['bbox']]
        add('scale-'+str(scale),ps,page=[400.*scale,600.*scale])
    ps=base()
    for p in ps:p['bbox']=[int(x)for x in p['bbox']]
    add('integer-coordinates',ps,page=[400,600])
    ps=base()
    for i,p in enumerate(ps):p['bbox']=[int(x)if (i+j)%2 else x for j,x in enumerate(p['bbox'])]
    add('mixed-coordinates',ps,page=[400,600.])
    rng=random.Random(261112)
    for i in range(100):
        ps=base(rows=rng.choice([3,4,5]),cols=rng.choice([2,3,4]));rng.shuffle(ps);p=rng.choice(ps);p['bbox'][rng.randrange(4)]+=rng.choice([-2.001,-2.,-.001,.001,2.,2.001]);add('permuted-'+str(i),ps)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_area,_bbox_overlap_in_smaller
    numbers=[]
    for a,b in [([0,0,500000001,499999999],[0,0,500000000,500000000]),([0,0,500000001.,499999999],[0,0,500000000,500000000]),([0,0,999999999,999999997],[100,200,999999998,999999996]),([10,20,100,200],[20.,20,100,200]),([0,0,2,2],[1,0,3,2]),([0,0,100,100],[100,0,200,100])]:
        aa=_bbox_area(tuple(a));bb=_bbox_area(tuple(b));numbers.append(dict(a=a,b=b,area_compare=(aa>bb)-(aa<bb),ratio=aa/bb,overlap=_bbox_overlap_in_smaller(tuple(a),tuple(b))))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_constants.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,numbers=numbers));print('Frozen',len(cases),'full filled-grid cases;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',len(numbers),'integer/mixed arithmetic facts')
if __name__=='__main__':main()
