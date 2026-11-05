"""Freeze full old diagram component detector including its actual table detection dependency."""
import copy,json,random
from common import ROOT,digest,write_json
from export_native_table_page_policies import source,empty
from export_supplemental_rule_tables import line

def path(b,segments=40,**kw):return dict(dict(bbox=b,segment_count=segments,form_depth=0,fill_visible=True,stroke_visible=False,rectangle_bboxes=[],source_index=0,fill_rgba=[180,60,50,255]),**kw)
def base():
    s=empty();s['chars']=[dict(char='中',bbox=[i*10.,0.,i*10.+10.,30.],tight_bbox=[i*10.,5.,i*10.+10.,15.])for i in range(20)];s['path_infos']=[path([60.,100.,90.,140.]),path([90.,100.,120.,140.]),path([120.,100.,150.,140.])];return dict(source=s,em=10.)
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_native_diagram_components
    return dict(diagram_component_bboxes=[list(b)for b in _detect_native_diagram_components(source(q['source']),q['em'])])
def main():
    target=ROOT/'build/migration/native-diagram-components.json';pdf_target=ROOT/'build/migration/pdf-diagram-components.json'
    if target.exists()or pdf_target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    def add(label,q):
        q=copy.deepcopy(q)
        try:out=reference(q)
        except (ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=out))
    add('positive',base());assert cases[0]['expected']['diagram_component_bboxes']
    for n in [0,1,19,20,21]:q=base();q['source']['chars']=q['source']['chars'][:n];add('anchors-'+str(n),q)
    for mode in ['empty','one-path','two-paths','small-segments','form','invisible','small-dot','outside-top','outside-bottom','large','prose','one-prose','near-photo','distant-photo','large-photo','photo-prose','two-round','int','missing-tight','no-positive-height']:
        q=base();s=q['source'];ps=s['path_infos']
        if mode=='empty':s['path_infos']=[]
        if mode=='one-path':s['path_infos']=ps[:1]
        if mode=='two-paths':s['path_infos']=ps[:2]
        if mode=='small-segments':
            for p in ps:p['segment_count']=33
        if mode=='form':
            for p in ps:p['form_depth']=1
        if mode=='invisible':
            for p in ps:p['fill_visible']=False
        if mode=='small-dot':
            for p in ps:p['segment_count']=13;p['bbox']=[60,100,70,110]
        if mode in ['outside-top','outside-bottom']:
            shift=-70 if mode=='outside-top'else 430
            for p in ps:p['bbox'][1]+=shift;p['bbox'][3]+=shift
        if mode=='large':ps[0]['bbox']=[0,50,490,550]
        if mode in ['prose','one-prose','photo-prose']:
            s['lines']=[line('These ordinary prose words cross the entire body',[50.,100.+i*15,155.,112.+i*15],i)for i in range(1 if mode=='one-prose'else 2)]
        if mode in ['near-photo','distant-photo','large-photo','photo-prose','two-round']:s['image_bboxes']=[[160.,105.,200.,145.]]
        if mode=='distant-photo':s['image_bboxes']=[[300.,300.,350.,350.]]
        if mode=='large-photo':s['image_bboxes']=[[160.,100.,490.,400.]]
        if mode=='two-round':s['image_bboxes'].append([225.,105.,250.,145.])
        if mode=='int':
            for p in ps:p['bbox']=[int(n)for n in p['bbox']]
        if mode=='missing-tight':
            for ch in s['chars']:ch['tight_bbox']=None
        if mode=='no-positive-height':s['lines']=[dict(line('x',[60.,100.,70.,110.],i),effective_height=0.)for i in range(4)]
        add(mode,q)
    for dist in [6.999999,7.,7.000001,14.999999,15.,15.000001]:
        q=base();q['source']['path_infos'][1]['bbox'][0]+=dist;q['source']['path_infos'][1]['bbox'][2]+=dist;add('connection-'+str(dist),q)
    for y in [47.9999,48.,48.0001,557.9999,558.,558.0001]:
        q=base();shift=y-q['source']['path_infos'][0]['bbox'][1]
        for p in q['source']['path_infos']:p['bbox'][1]+=shift;p['bbox'][3]+=shift
        add('margin-'+str(y),q)
    for ink in [4.9999,5.,14.,14.0001]:
        q=base()
        for ch in q['source']['chars']:ch['tight_bbox'][3]=5.+ink
        add('ink-'+str(ink),q)
    for mode in ['timeline','timeline-prose','timeline-no-strip','timeline-gray','timeline-short','timeline-deep-label']:
        q=base();s=q['source'];s['path_infos']=[]
        for j in range(4):
            for k in range(3):s['path_infos'].append(path([40.+j*100+k*10,100.,50.+j*100+k*10,130.]))
        if mode!='timeline-no-strip':s['path_infos'].append(path([40.,150.,370.,160.],segments=5,rectangle_bboxes=[[40.,150.,370.,160.]],fill_rgba=[250,250,250,255]if mode=='timeline-gray'else[20,80,180,255]))
        if mode=='timeline-short':s['path_infos'][-1]['bbox'][2]=100.
        s['lines']=[line('Label',[40.+j*100,190.,70.+j*100,200.],j)for j in range(4)]
        if mode=='timeline-deep-label':s['lines'][-1]['bbox']=[340.,360.,370.,370.]
        if mode=='timeline-prose':s['lines']+=[line('These ordinary words form a barrier across icons',[40.,100.+i*15,370.,110.+i*15],10+i)for i in range(2)]
        add(mode,q)
    rng=random.Random(271028)
    for i in range(100):
        q=base();s=q['source'];s['path_infos']=[path([x,y,x+rng.choice([10.,25.,40.]),y+rng.choice([10.,30.,45.])],rng.choice([13,20,33,40,60]))for x,y in [(rng.choice([40.,60.,90.,120.,180.,220.]),rng.choice([70.,100.,130.,160.]))for _ in range(rng.randrange(3,12))]]
        add('random-'+str(i),q)
    # True table barriers: provide real tight glyphs so old inflated caption binding succeeds.
    old=json.loads((ROOT/'build/migration/native-complete-table-candidates.json').read_text(encoding='utf-8'))
    for c in old['cases'][::23]:
        q=base();raw=copy.deepcopy(c['request']['source']);raw['chars']=q['source']['chars'];raw.setdefault('form_bboxes',[])
        for l in raw.get('lines',[]):l['chars']=[dict(tight_bbox=l['bbox'])]
        raw['path_infos']=raw.get('path_infos',[])+q['source']['path_infos'];q['source']=raw;add('table-barrier-'+c['label'],q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['graphics.py','isolated_graphics.py','table_detection.py','table_filled_grid.py','geometry.py']]
    meta=dict(sources={p:digest((ROOT/p).read_bytes())for p in refs});write_json(target,dict(meta,scope='Actual complete old native diagram component function, including full table detector; synthetic raw source, not injected table verdicts.',cases=cases,errors=errors))
    pdf=[]
    for c in json.loads((ROOT/'build/migration/pdf-complete-table-candidates.json').read_text(encoding='utf-8'))['cases']:
        q=dict(source=c['request']['source'],em=10.);pdf.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],request=q,expected=reference(q)))
    write_json(pdf_target,dict(meta,scope='55 actual PDF sources; detector may legitimately exit at inflated CJK gate, no actual positive claim.',cases=pdf));print('Frozen diagrams',len(cases),'errors',len(errors),'positive',sum(bool(c['expected']['diagram_component_bboxes'])for c in cases),'PDF',len(pdf))
if __name__=='__main__':main()
