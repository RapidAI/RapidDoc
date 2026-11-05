"""Freeze original page policies and the actual complete old candidate detector; never overwrite."""
import copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects as lines
from export_native_table_candidate_merge import objects,candidate,annotation
from export_native_tiled_tables import old_source
from export_closed_rule_pipeline import dumped
from export_supplemental_rule_tables import line

def source(raw):
    s=old_source(raw);s.lines=lines(raw.get('lines',[]))
    for k,default in [('form_bboxes',[]),('image_bboxes',[]),('signature_bboxes',[]),('page_index',None),('previous_table_grid',None)]:setattr(s,k,raw.get(k,default))
    if s.previous_table_grid is not None:s.previous_table_grid=SimpleNamespace(**s.previous_table_grid)
    return s

def reference(q,complete=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_detection as t
    s=source(q['source'])
    if complete:cs=t._detect_table_candidates(s,q.get('excluded_bboxes',[]))
    else:
        cs=objects(q['candidates']);op=q.get('operation','all')
        if op in ['all','join']:cs=t._join_caption_supported_table_groups(s,cs)
        if op in ['all','trim']:
            for c in cs:t._trim_unruled_table_prose_tail(s,c)
        if op in ['all','forms']:
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_first
            cs=[c for c in cs if not any(_bbox_overlap_in_first(c.core_bbox or c.bbox,f)>=.9 for f in s.form_bboxes)]
        if op in ['all','externalize']:t._externalize_table_continuation_captions(s,cs)
        if op in ['all','bind']:t._bind_short_native_grid_captions(s,cs)
        if op=='all':cs.sort(key=lambda c:(c.bbox[1],c.bbox[0]))
    return clean(dict(candidates=[dumped(c)for c in cs]))

def empty():return dict(page_size=[500.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],form_bboxes=[],image_bboxes=[],signature_bboxes=[])
def base(op='join'):
    s=empty();s['lines']=[line('Table 1: Results',[40.,225.,240.,235.],90)]
    return dict(source=s,candidates=[candidate([40.,100.,240.,150.]),candidate([40.,160.,240.,220.],ids=[2])],operation=op)
def main():
    targets=[ROOT/f'build/migration/{n}.json'for n in ['native-table-page-policies','native-complete-table-candidates','pdf-complete-table-candidates']]
    if any(t.exists()for t in targets[1:]):raise FileExistsError(targets)
    cases=[];errors=[]
    def add(label,q):
        q=clean(copy.deepcopy(q))
        try:expected=reference(q)
        except (ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=expected))
    add('join',base())
    for text in ['Table 1: Results','tab. A1: Results','表１：结果','Table İ1. Example','Table ١: Results','Table 1 Results','Table 1:','Table1:','表格1:','table K.1:','Table A.1:']:
        q=base();q['source']['lines'][0]['text']=text;add('caption-'+text,q)
    for gap in [14.9999,15.,15.0001,30.]:
        q=base();q['candidates'][1]['bbox'][1]=150.+gap;q['candidates'][1]['local_bbox'][1]=150.+gap;add('join-gap-'+str(gap),q)
    for mode in ['barrier','header','italic','annotation','rules','no-caption','rotated','sparse','sparse-prose','sparse-semantic','sparse-two-caption']:
        q=base();s=q['source']
        if mode=='barrier':s['lines'].append(line('Table 2: Other',[40.,151.,240.,159.],91))
        if mode in ['header','italic']:s['lines']+=[line('Header',[40.,80.,100.,95.],1)]
        if mode=='header':s['lines'].append(line('Column',[160.,80.,240.,95.],2))
        if mode=='italic':s['lines'][-1]['font_signature']=['Serif',64]
        if mode=='annotation':q['candidates'][0]['annotations']=[annotation()]
        if mode=='rules':s['drawing_lines']=[dict(bbox=[39,99,241,99],orientation='horizontal',width=0)]
        if mode=='no-caption':s['lines']=[]
        if mode=='rotated':q['candidates'][0]['angle']=90
        if mode.startswith('sparse'):
            s['drawing_lines']=[dict(bbox=[40,y,240,y],orientation='horizontal',width=0)for y in [100,120,140,170,190,220]]
            s['lines'].append(line('Label',[50,180,90,190],4))
            if mode=='sparse-prose':s['lines'][-1].update(text='These many ordinary words prevent merging a prose body region',bbox=[40,180,240,190])
            if mode=='sparse-semantic':s['lines'][-1]['semantic_type']='caption'
            if mode=='sparse-two-caption':s['lines'].insert(0,line('Table 0: Before',[40.,80.,240.,90.],89))
        add(mode,q)
    rng=random.Random(271027)
    for n in [5,16,64,100,160]:
        q=base();q['candidates']=[candidate([40.,100.+(j%12)*10.,240.,108.+(j%12)*10.],ids=[j],score=j)for j in range(n)];rng.shuffle(q['candidates']);add('pending-'+str(n),q)
    q=dict(source=empty(),candidates=[candidate([40,100,240,260],ids=range(12))],operation='trim');s=q['source']
    for i in range(3):s['lines'] += [line('10',[45,110+20*i,70,120+20*i],i*2),line('20',[180,110+20*i,210,120+20*i],i*2+1)]
    for i in range(3):s['lines'].append(dict(line('These ordinary words form a distinct prose paragraph below data',[40,175+25*i,240,187+25*i],6+i),effective_height=12.))
    add('trim-positive',q)
    for mode in ['note','vertical','horizontal','short-tail','small-font','angle','narrow','annotation','int-float']:
        v=copy.deepcopy(q);s=v['source']
        if mode=='note':s['lines'][6]['text']='Note: These ordinary words form a distinct prose paragraph'
        if mode=='vertical':s['drawing_lines']=[dict(bbox=[40,100,40,260],orientation='vertical',width=0)]
        if mode=='horizontal':s['drawing_lines']=[dict(bbox=[40,200,240,200],orientation='horizontal',width=0)]
        if mode=='short-tail':s['lines']=s['lines'][:-1]
        if mode=='small-font':
            for l in s['lines'][6:]:l['effective_height']=10.
        if mode=='angle':v['candidates'][0]['angle']=180
        if mode=='narrow':v['candidates'][0]['bbox'][2]=100;v['candidates'][0]['local_bbox'][2]=100
        if mode=='annotation':v['candidates'][0]['annotations']=[annotation()]
        if mode=='int-float':v['candidates'][0]['local_bbox']=[40.,100,240.,260]
        add('trim-'+mode,v)
    for overlap in [89.9999,90.,90.0001,100.]:
        q=dict(source=empty(),candidates=[candidate([0,0,100,100])],operation='forms');q['source']['form_bboxes']=[[0,0,overlap,100]];add('forms-'+str(overlap),q)
    q=dict(source=empty(),candidates=[candidate([40.,80.,240.,220.],ids=[1,2])],operation='externalize');s=q['source'];s['lines']=[line('Table 1 continued',[40.,80.,240.,90.],1)]+[line('data',[50.,110.+30*i,80.,120.+30*i],2+i)for i in range(3)];s['drawing_lines']=[dict(bbox=[40.,y,240.,y],orientation='horizontal',width=0.)for y in [100.,140.,180.,220.]]+[dict(bbox=[x,100.,x,220.],orientation='vertical',width=0.)for x in [40.,140.,240.]]
    for text in ['Table 1 continued','Table continued','tab. IV cont.','续表','续 表 １','续表十','TABLE 12CONTINUED','Table 1 cont.','Table ١ continued','续表壹','continued','Table I continued']:
        v=copy.deepcopy(q);v['source']['lines'][0]['text']=text;add('continuation-'+text,v)
    for mode in ['existing-caption','duplicate-caption','duplicate-id','annotation-order','no-grid','integer']:
        v=copy.deepcopy(q)
        if mode in ['existing-caption','annotation-order','duplicate-id']:v['candidates'][0]['annotations']=[annotation(ids=[8,9])]
        if mode=='annotation-order':v['candidates'][0]['annotations'].insert(0,annotation('note'))
        if mode=='duplicate-id':v['source']['lines'][0]['source_index']=8
        if mode=='duplicate-caption':v['source']['lines'].append(line('续表',[40.,78.,240.,88.],99))
        if mode=='no-grid':v['source']['drawing_lines']=[]
        if mode=='integer':
            for l in v['source']['lines']:l['bbox']=[int(x)for x in l['bbox']]
        add('externalize-'+mode,v)
    q=dict(source=empty(),candidates=[candidate([40.,100.,340.,220.])],operation='bind');s=q['source'];s['chars']=[dict(char='中',bbox=[i*10.,0.,i*10.+10.,30.],tight_bbox=[i*10.,5.,i*10.+10.,15.])for i in range(20)];s['lines']=[dict(line('中文表格独立题名',[40.,80.,120.,90.],6),chars=[dict(tight_bbox=[40.,80.,120.,90.])])]
    add('bind-positive',q)
    for mode in ['peer','rotated-peer','duplicate-line-id','owned','nineteen','tight','short','long','too-far','core','missing-tight']:
        v=copy.deepcopy(q);s=v['source']
        if mode in ['peer','rotated-peer','duplicate-line-id']:s['lines'].append(dict(line('Other',[180.,80.,220.,90.],6 if mode=='duplicate-line-id'else 7),angle=90 if mode=='rotated-peer'else 0,chars=[dict(tight_bbox=[180.,80.,220.,90.])]))
        if mode=='owned':v['candidates'][0]['annotations']=[dict(kind='caption',bbox=[40.,80.,120.,90.],line_indices=[6],line_bboxes=[[6,[40.,80.,120.,90.]]])]
        if mode=='nineteen':s['chars']=s['chars'][:19]
        if mode=='tight':s['lines'][0]['chars']=[dict(tight_bbox=[40,82,120,95])]
        if mode=='short':s['lines'][0]['text']='中文短表名'
        if mode=='long':s['lines'][0]['text']='中'*36
        if mode=='too-far':s['lines'][0]['bbox'][3]=85.
        if mode=='core':v['candidates'][0]['core_bbox']=[40,105,340,220]
        if mode=='missing-tight':s['lines'][0]['chars']=[]
        add('bind-'+mode,v)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['table_detection.py','table_rules.py','table_filled_grid.py','table_annotations.py','geometry.py','models.py']];meta=dict(sources={p:digest((ROOT/p).read_bytes())for p in refs})
    policy_data=dict(meta,scope='Individual final page policies; explicit candidate projections, exact old helpers.',cases=cases,errors=errors)
    if targets[0].exists():assert json.loads(targets[0].read_text(encoding='utf-8'))==policy_data,'existing immutable policy reference differs'
    else:write_json(targets[0],policy_data)
    complete=[];complete_errors=[]
    def full(label,q):
        try:expected=reference(q,True)
        except (ValueError,KeyError)as e:complete_errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        complete.append(dict(label=label,request=q,expected=expected))
    old=json.loads((ROOT/'build/migration/native-all-candidate-producers.json').read_text(encoding='utf-8'))
    for c in old['cases']:
        q=c['request'];full(c['label'],q)
    for c in cases:
        q=dict(source=c['request']['source'],excluded_bboxes=[]);full('policy-source-'+c['label'],q)
    write_json(targets[1],dict(meta,scope='Actual original _detect_table_candidates, all producers and final policies.',cases=complete,errors=complete_errors))
    pdf=[]
    for name in ['pdf-final-candidate-producers','pdf-aligned-final-positive']:
        for c in json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']:
            c=copy.deepcopy(c);c['expected']=reference(c['request'],True);pdf.append(c)
    write_json(targets[2],dict(meta,scope='Actual original complete detector on frozen actual PDF source; 55 pages, fixed-font CID excluded.',cases=pdf))
    print('Frozen',len(cases),'policy,',len(complete),'complete,',len(pdf),'actual PDF page references')
if __name__=='__main__':main()
