"""Original full document geometry generator; only raw lines and geometry are supplied."""
import copy,json,random
from common import ROOT,write_json,digest
from export_native_geometry import lines,plan_raw
from export_native_late_image_text import project
from export_native_table_page_policies import empty

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.char_geometry import build_document_geometry_plan
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPageTextGeometry
    ls=[];gs=[];sizes=[]
    for p in q['pages']:
        s=p['source'];ls.append(lines(s['lines']));raw=p['geometry'];maps=lambda k,value:{r['char_idx']:tuple(r[value])for r in raw.get(k,[])};gs.append(PDFPageTextGeometry(chars=copy.deepcopy(raw.get('chars',[])),tight_bboxes=maps('tight_bboxes','bbox'),origins=maps('origins','origin'),loose_bboxes=maps('loose_bboxes','bbox')));sizes.append(tuple(s['page_size']))
    plan=build_document_geometry_plan(ls,gs,sizes)
    return dict(plan=plan_raw(plan),remaining_loose_bboxes=[[dict(char_idx=i,bbox=list(b))for i,b in sorted(g.loose_bboxes.items())]for g in gs])
def page(rows=4,n=20,advance=7.,cell=7.,height=10.,tight_height=8.,font=10.,spacing=20.,name='Body',angle=0):
    source=empty();source['page_size']=[600.,900.];geometry=dict(chars=[],tight_bboxes=[],origins=[],loose_bboxes=[])
    for j in range(rows):
        baseline=120.+j*spacing;chars=[]
        for i in range(n):
            idx=j*n+i;x=40.+i*advance;box=[x,baseline-height+2.,x+cell,baseline+2.];tight=[x,baseline-tight_height,x+5.,baseline];origin=[x,baseline];text='A'if i%5 else'2';chars.append(dict(char=text,char_idx=idx,bbox=box,rotation=0.,font=dict(name=name,size=font,flags=0,weight=400)))
            geometry['tight_bboxes'].append(dict(char_idx=idx,bbox=tight));geometry['origins'].append(dict(char_idx=idx,origin=origin));geometry['loose_bboxes'].append(dict(char_idx=idx,bbox=box))
        source['lines'].append(dict(text=''.join(c['char']for c in chars),bbox=[40.,baseline-height+2.,40.+(n-1)*advance+cell,baseline+2.],angle=angle,source_index=j,chars=chars,effective_height=height))
        geometry['chars']+=chars
    source['chars']=copy.deepcopy(geometry['chars']);return dict(source=source,geometry=geometry)
def rotate(p,a):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    size=tuple(p['source']['page_size']);world=lambda b:list(_rotate_bbox_from_upright(tuple(b),size,a))
    point=lambda o:[size[0]-o[1],o[0]]if a==90 else [o[1],size[1]-o[0]]if a==270 else[size[0]-o[0],size[1]-o[1]]if a==180 else o
    for l in p['source']['lines']:
        l['angle']=a;l['bbox']=world(l['bbox'])
        for c in l['chars']:c['bbox']=world(c['bbox'])
    for k in ['tight_bboxes','loose_bboxes']:
        for r in p['geometry'][k]:r['bbox']=world(r['bbox'])
    for r in p['geometry']['origins']:r['origin']=point(r['origin'])
    p['source']['chars']=[copy.deepcopy(c)for l in p['source']['lines']for c in l['chars']];p['geometry']['chars']=copy.deepcopy(p['source']['chars'])
    return p
def main():
    target=ROOT/'build/migration/native-geometry-plans.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,pages):
        q=copy.deepcopy(dict(pages=pages))
        for p in q['pages']:p['source']['lines']=[project(l)for l in lines(p['source']['lines'])]
        cases.append(dict(label=label,request=q,expected=reference(q)))
    add('empty',[])
    for mode in ['healthy','no-tight','no-origins','one-map-only','x-bad','x-short','y-bad','style-only','style-one-page','mixed','split-shadow','skip-y-formula','skip-y-cluster','side-map-rotation','side-map-overwide','raw-provenance','small-anchors','unicode','font-fields','duplicate-source','duplicate-chars','local-extreme','local-too-few','y-healthy-neighbor','loose-tiers']:
        p=page();pages=[p]
        if mode=='no-tight':p['geometry']['tight_bboxes']=[]
        if mode=='no-origins':p['geometry']['origins']=[]
        if mode=='one-map-only':p['geometry']['tight_bboxes']=[];p2=page();p2['geometry']['origins']=[];pages.append(p2)
        if mode=='x-bad':pages=[page(cell=14.)]
        if mode=='x-short':pages=[page(rows=1,n=20,cell=14.)]
        if mode=='y-bad':pages=[page(rows=6,height=40.,spacing=14.)]
        if mode=='style-only':pages=[page(rows=3,height=16.,spacing=40.),page(rows=3,height=16.,spacing=40.)]
        if mode=='style-one-page':pages=[page(rows=6,height=16.,spacing=40.)]
        if mode=='mixed':pages=[page(height=40.,cell=14.,spacing=14.),page(height=40.,cell=14.,spacing=14.)]
        if mode=='split-shadow':
            for l in p['source']['lines']:
                for c in l['chars'][10:]:
                    i=c['char_idx'];p['geometry']['origins'][i]['origin'][1]+=14.;p['geometry']['tight_bboxes'][i]['bbox'][1]+=14.;p['geometry']['tight_bboxes'][i]['bbox'][3]+=14.
        if mode in ['skip-y-formula','skip-y-cluster']:
            p=page(height=40.,spacing=14.);pages=[p]
            for l in p['source']['lines']:l['formula_candidate_only'if mode=='skip-y-formula'else'restored_inline_cluster']=True
        if mode.startswith('side-map'):
            pages=[page(cell=14.)];p=pages[0]
            for l in p['source']['lines']:
                for c in l['chars']:c['rotation']=90.
            for r in p['geometry']['loose_bboxes']:
                r['bbox'][2]+=100. if mode.endswith('overwide')else 2.
        if mode=='raw-provenance':
            p=page(height=40.,spacing=14.);pages=[p]
            for l in p['source']['lines']:l['bbox'][1]+=4.;l['bbox'][3]-=4.
        if mode=='small-anchors':
            p=page(cell=14.);pages=[p]
            for r in p['geometry']['tight_bboxes']:
                if r['char_idx']%3==0:r['bbox'][1]=r['bbox'][3]-1.
        if mode=='unicode':
            p=page(cell=14.);pages=[p];letters='中あ한αΩⅫ²ﬃA２'
            for l in p['source']['lines']:
                for i,c in enumerate(l['chars']):c['char']=letters[i%len(letters)]
        if mode=='font-fields':
            p=page(cell=14.);pages=[p]
            for l in p['source']['lines']:
                for c in l['chars']:c['font']=dict(name='ABCDEF+B_o-d y',size='10',flags='0',weight='450')
        if mode=='duplicate-source':
            p=page(height=40.,cell=14.,spacing=14.);pages=[p];p['source']['lines'][2]['source_index']=0
        if mode=='duplicate-chars':
            p=page(cell=14.);pages=[p];p['source']['lines'][0]['chars'][1]['char_idx']=0
        if mode in ['local-extreme','local-too-few']:
            p=page(rows=6 if mode=='local-extreme'else 5,n=16,height=40.,font=1.,spacing=18.,name='BodyRegular');donor=page(rows=2,n=16,height=12.8,font=1.,spacing=20.,name='BodyBold')
            shift=1000;line_shift=len(p['source']['lines'])
            for l in donor['source']['lines']:
                l['source_index']+=line_shift;l['bbox'][1]+=300.;l['bbox'][3]+=300.
                for c in l['chars']:c['char_idx']+=shift;c['bbox'][1]+=300.;c['bbox'][3]+=300.
            for key in ['tight_bboxes','loose_bboxes','origins']:
                for r in donor['geometry'][key]:r['char_idx']+=shift;v=r['origin'if key=='origins'else'bbox'];v[1]+=300.;v[3:]=[x+300. for x in v[3:]]
                p['geometry'][key]+=donor['geometry'][key]
            p['source']['lines']+=donor['source']['lines'];pages=[p]
        if mode=='y-healthy-neighbor':
            p=page(rows=8,height=40.,spacing=14.);pages=[p]
            for l in p['source']['lines'][5:]:
                l['effective_height']=10.;l['bbox'][1]=l['bbox'][3]-10.
                for c in l['chars']:c['bbox'][1]=c['bbox'][3]-10.
        if mode=='loose-tiers':
            p=page(rows=6,height=40.,spacing=14.);pages=[p]
            for l in p['source']['lines']:
                for c in l['chars'][:10]:c['bbox'][1]=c['bbox'][3]-16.
        add(mode,pages)
    for a in [90,180,270]:add(f'rotation/{a}',[rotate(page(cell=14.),a)])
    rng=random.Random(103)
    for i in range(45):
        pages=[page(rows=rng.choice([1,2,3,6]),n=rng.choice([5,12,20,35]),cell=rng.choice([7.,8.1,9.1,9.46,12.,14.]),height=rng.choice([10.,15.,16.,17.6,24.,40.]),font=rng.choice([1.,8.,10.]),spacing=rng.choice([14.,25.,45.]))for _ in range(rng.choice([1,1,2]))];add(f'geometry/{i}',pages)
    actual=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text(encoding='utf-8'));pages=[]
    for s in actual['request']['sources']:
        chars=s['chars'];rows=lambda k,v:[dict(char_idx=c['char_idx'],**{v:c[k]})for c in chars if c.get(k)is not None];pages.append(dict(source=s,geometry=dict(chars=chars,tight_bboxes=rows('tight_bbox','bbox'),origins=rows('origin','origin'),loose_bboxes=rows('loose_bbox','bbox'))))
    add('actual-form-document',pages)
    write_json(target,dict(sources={'char_geometry.py':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/char_geometry.py').read_bytes())},cases=cases));print(len(cases),'complete old document plans;',sum(bool(c['expected']['plan']['char_repairs'])for c in cases),'char repair documents;',sum(bool(c['expected']['plan']['line_repairs'])for c in cases),'line repair documents')
if __name__=='__main__':main()
