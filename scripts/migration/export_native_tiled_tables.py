"""Freeze full original tiled-fill candidate output, never overwrite reference."""
import copy,random
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def old_source(s):
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    return SimpleNamespace(page_size=s['page_size'],chars=s.get('chars',[]),path_infos=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox']),rectangle_bboxes=tuple(tuple(b)for b in p['rectangle_bboxes'])))for p in s.get('path_infos',[])],drawing_lines=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in s.get('drawing_lines',[])],lines=[SimpleNamespace(**l)for l in s.get('lines',[])])

def reference(s,excluded=()):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_tiled_fill_table_candidates
    return clean(dict(candidates=[asdict(c)for c in _detect_tiled_fill_table_candidates(old_source(s),excluded)]))

def base(y=100.,header=False):
    chars=[dict(char='中',bbox=[i*10.,0.,i*10.+10.,30.],tight_bbox=[i*10.,5.,i*10.+10.,15.])for i in range(20)]
    paths=[]
    for row in range(3):
        for col in range(2):
            b=[50.+col*100.,y+row*30.,150.+col*100.,y+(row+1)*30.]
            paths.append(dict(bbox=b,rectangle_bboxes=[b],fill_visible=True,stroke_visible=False,segment_count=5,form_depth=0,source_index=0))
    rules=[dict(bbox=[150.,y,150.,y+90.],width=1.,orientation='vertical')]+[dict(bbox=[50.,y+i*30.,250.,y+i*30.],width=1.,orientation='horizontal')for i in [1,2]]
    lines=[dict(text='正文',bbox=[60.,y+10.+row*30.,80.,y+20.+row*30.],source_index=row,chars=[])for row in range(3)]
    if header:
        lines.extend(dict(text='表头',bbox=[60.+i*100.,y-15.,80.+i*100.,y-5.],source_index=3+i,chars=None)for i in range(2))
    return dict(page_size=[400.,600.],chars=chars,path_infos=paths,drawing_lines=rules,lines=lines)

def main():
    target=ROOT/'build/migration/native-tiled-tables.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,s=None,excluded=()):
        s=copy.deepcopy(s or base());cases.append(dict(label=label,request=clean(dict(source=s,excluded_bboxes=excluded)),expected=reference(s,excluded)))
    add('plain');add('header',base(header=True))
    for count in [0,1,19,20,21]:
        s=base();s['chars']=s['chars'][:count];add('anchors-'+str(count),s)
    for loose in [21.999,22.,22.001,30.]:
        for ink in [4.999,5.,5.001,14.,14.001]:
            s=base()
            for c in s['chars']:c['bbox'][3]=loose;c['tight_bbox'][3]=5.+ink
            add(f'metrics-{loose}-{ink}',s)
    for key,values in [('form_depth',[0,1]),('segment_count',[4,5,6]),('fill_visible',[False,True])]:
        for value in values:
            s=base()
            for p in s['path_infos']:p[key]=value
            add(key+str(value),s)
    for count in [0,1,2,5,6]:
        s=base();s['path_infos']=s['path_infos'][:count];add('members-'+str(count),s)
    for gap in [.49999,.5,.50001]:
        s=base()
        for p in s['path_infos'][2:]:p['bbox'][1]+=gap
        add('touch-'+str(gap),s)
    for exclude in [[50.,100.,150.,190.],[50.,100.,100.,145.],[300.,300.,350.,350.]]:add('excluded-'+str(exclude),excluded=[exclude])
    for count in [0,1,2,3]:
        s=base();s['drawing_lines']=s['drawing_lines'][:count];add('rules-'+str(count),s)
    for ext in [26.999,27.,27.001,60.]:
        s=base();s['drawing_lines'][0]['bbox'][3]+=ext;add('extend-'+str(ext),s)
    for shift in [0.,.999,1.,1.001,3.]:
        s=base(header=True);s['lines'][-1]['bbox'][1]+=shift;add('header-slack-'+str(shift),s)
    for chars in [[],None,[dict(tight_bbox=[60.,110.,80.,120.])]]:
        s=base(header=True);s['lines'][-1]['chars']=chars;add('ink-'+str(chars),s)
    for count in [16,17]:
        s=base(header=True)
        for l in s['lines'][-2:]:l['text']='中'*count
        add('header-length-'+str(count),s)
    # The preceding grid extends through a short title; next grid owns the annotation.
    def pair(caption=True):
        s=base();t=base(y=220.,header=True)
        s['drawing_lines'][0]['bbox'][3]=215.
        for l in t['lines']:l['source_index']+=10
        s['path_infos']+=t['path_infos'];s['drawing_lines']+=t['drawing_lines'];s['lines']+=t['lines']
        if caption:s['lines'].append(dict(text='下一表独立完整题名',bbox=[50.,199.,130.,209.],source_index=30,chars=[]))
        return s
    add('pair-caption',pair());add('pair-no-caption',pair(False))
    s=pair();s['lines'][-1]['source_index']=0;add('duplicate-source-caption',s)
    s=base(header=True)
    for l in s['lines']:l['source_index']=-l['source_index']-1
    add('signed-sources',s)
    s=base();s['path_infos']*=2;add('duplicates',s)
    rng=random.Random(261109)
    for i in range(100):
        s=base(header=bool(i%2));rng.shuffle(s['path_infos']);rng.shuffle(s['drawing_lines']);rng.shuffle(s['lines'])
        p=rng.choice(s['path_infos']);p['bbox'][rng.randrange(4)]+=rng.choice([-.50001,-.5,-.01,0.,.01,.5,.50001]);add('shuffle-geometry-'+str(i),s)
    # Independent CPython set traces cover resizing after difference_update and pop finger.
    traces=[]
    for n in [6,17,33,65,129,513,2050,5000]:
        pending=set(range(n));steps=[]
        while pending:
            seed=pending.pop();order=list(pending);remove=[k for k in order if rng.randrange(5)<4];pending.difference_update(remove);steps.append(dict(seed=seed,order=order,remove=remove,remaining=list(pending)))
        traces.append(dict(count=n,steps=steps))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,set_traces=traces))
    print('Frozen',len(cases),'old tiled-table cases;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',sum(any(x['annotations']for x in c['expected']['candidates'])for c in cases),'caption transfers')
if __name__=='__main__':main()
