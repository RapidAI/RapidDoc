"""Full-generator boundary scenes: sparse donors, provenance, tiers, and local metrics."""
import copy,json
from common import ROOT,write_json
from export_native_geometry_plans import page,reference
from export_native_geometry import lines
from export_native_late_image_text import project
def main():
    target=ROOT/'build/migration/native-geometry-plan-edges.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,pages):
        q=dict(pages=copy.deepcopy(pages))
        for p in q['pages']:p['source']['lines']=[project(l)for l in lines(p['source']['lines'])]
        cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['sparse-tail','sparse-no-adjacent','sparse-column','sparse-small-overhang','source-x-provenance','source-y-provenance','invalid-tight','missing-origin','generated-space','backward-origin','empty-font','unknown-font','subset-prefix','unicode-font','font-rounding','style-healthy-page','style-duplicate-source','baseline-shadow','mixed-anchor-sizes','healthy-offsets','tier-offsets','tiny-style']:
        p=page(rows=6,cell=14.);pages=[p]
        if mode.startswith('sparse-'):
            for l in p['source']['lines']:
                pos=9 if mode=='sparse-column'else 19;c=l['chars'][pos];c['char']='β';c['bbox'][2]+=100. if mode!='sparse-small-overhang'else 2.
                if mode=='sparse-no-adjacent':l['chars'][pos-1]['font']['flags']=1
                if mode=='sparse-column':
                    c['bbox'][2]+=150.
                    for after in l['chars'][pos+1:]:
                        i=after['char_idx'];after['bbox'][0]+=200.;after['bbox'][2]+=200.;p['geometry']['origins'][i]['origin'][0]+=200.;p['geometry']['tight_bboxes'][i]['bbox'][0]+=200.;p['geometry']['tight_bboxes'][i]['bbox'][2]+=200.
                l['bbox'][2]=max(x['bbox'][2]for x in l['chars'])
        if mode=='source-x-provenance':
            for l in p['source']['lines']:l['bbox'][2]-=2.
        if mode in ['source-y-provenance','baseline-shadow','mixed-anchor-sizes','healthy-offsets','tier-offsets']:
            p=page(rows=8,height=40.,spacing=14.);pages=[p]
            if mode=='source-y-provenance':
                for l in p['source']['lines']:l['bbox'][1]+=2.;l['bbox'][3]-=2.
            if mode=='baseline-shadow':
                for r in p['geometry']['origins']:
                    if r['char_idx']%20<8:r['origin'][1]+=15.
            if mode=='mixed-anchor-sizes':
                for r in p['geometry']['tight_bboxes']:
                    if r['char_idx']%20<4:r['bbox'][1]=r['bbox'][3]-1.
            if mode=='healthy-offsets':
                for l in p['source']['lines'][5:]:
                    l['bbox'][1]=l['bbox'][3]-12.;l['effective_height']=12.
                    for c in l['chars']:c['bbox'][1]=c['bbox'][3]-12.
            if mode=='tier-offsets':
                for l in p['source']['lines']:
                    for c in l['chars'][:8]:c['bbox'][1]=c['bbox'][3]-16.
        if mode=='invalid-tight':
            for r in p['geometry']['tight_bboxes'][::4]:r['bbox'][2]=r['bbox'][0]
        if mode=='missing-origin':p['geometry']['origins']=p['geometry']['origins'][::2]
        if mode=='generated-space':
            for l in p['source']['lines']:
                for c in l['chars'][::4]:c['char']=' ';c['is_generated']=True
        if mode=='backward-origin':
            for r in p['geometry']['origins'][::4]:r['origin'][0]-=20.
        if mode in ['empty-font','unknown-font','subset-prefix','unicode-font','font-rounding','tiny-style']:
            for l in p['source']['lines']:
                for c in l['chars']:
                    if mode=='empty-font':c['font']={}
                    if mode=='unknown-font':c['font']=None
                    if mode=='subset-prefix':c['font']['name']='ABCDEF+B_o-d y'
                    if mode=='unicode-font':c['font']['name']='ABCDEF+Straße İ'
                    if mode=='font-rounding':c['font'].update(size=10.125,weight=450.)
                    if mode=='tiny-style':c['font']['size']=0.25
        if mode in ['style-healthy-page','style-duplicate-source']:
            pages=[page(rows=3,height=16.),page(rows=3,height=16.),page(rows=3,height=10.)]
            if mode=='style-duplicate-source':
                for p in pages:p['source']['lines'][2]['source_index']=0
        add(mode,pages)
    # Boundary counts and ratios are consumed by the full producer, not directly assigned as run results.
    for n in [9,10,29,30,31]:
        for cell in [9.1,9.45,9.4500001,14.]:add(f'pair-boundary/{n}/{cell}',[page(rows=2,n=n,cell=cell)])
    write_json(target,dict(cases=cases));print(len(cases),'full producer boundary scenes;',sum(bool(c['expected']['plan']['char_repairs'])for c in cases),'char repair documents')
if __name__=='__main__':main()
