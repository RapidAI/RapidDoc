"""Freeze complete original early image/parallel-field/page-footnote sequence."""
import copy
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_image_chain import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.auxiliary_text import _classify_image_footnotes,_classify_page_footnotes
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.native_boundaries import protect_parallel_table_end_fields
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    ls=lines(q['lines']);sz=tuple(q['page_size']);images=[tuple(b)for b in q['visual_bboxes']];tables=[tuple(b)for b in q['table_bboxes']];draw=[_AxisLine(**dict(r,bbox=tuple(r['bbox'])))for r in q['drawing_lines']];refs=lines(q['reference_lines'])
    _classify_image_footnotes(ls,images,tables,draw,sz,reference_body_height=q.get('reference_body_height'))
    protect_parallel_table_end_fields(ls,tables)
    groups=_classify_page_footnotes(ls,tables,draw,sz,visual_bboxes=images,reference_lines=refs)
    return dict(lines=[project(l)for l in ls],page_footnote_groups=[sorted(g)for g in groups])

def base():
    ls=[line('Original complete native body paragraph',[30.,y,370.,y+12.],i,font_coverage=1.)for i,y in enumerate([150.,180.,210.,240.,270.])]
    for l in ls:l['effective_height']=12.
    ls += [dict(line('1. Original native note begins here',[30.,470.,250.,478.],10),effective_height=8.),dict(line('Original note continuation words',[30.,482.,250.,490.],11),effective_height=8.)]
    return dict(lines=ls,page_size=[400.,600.],visual_bboxes=[],table_bboxes=[],drawing_lines=[dict(bbox=[30.,460.,150.,460.],orientation='horizontal',width=.5)],reference_lines=[])

def scenes():
    cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['lines']=[project(l)for l in lines(q['lines'])];q['reference_lines']=[project(l)for l in lines(q.get('reference_lines',[]))];cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['ruled','unruled','empty','no-body','regular-size','far','caption','table','table-fragment','fraction','visual-overlap','short','tiny','column','column-middle','column-regular','fullwidth','duplicate-rules','lower-barrier','edge-marker','split-visual-row','restored-cluster','style-height','footer-tail','unicode-marker']:
        q=base()
        if mode=='unruled':q['drawing_lines']=[]
        if mode=='empty':q['lines']=[]
        if mode=='no-body':q['lines']=q['lines'][-2:]
        if mode=='regular-size':
            for l in q['lines'][-2:]:l['effective_height']=12.
        if mode=='far':
            for l in q['lines'][-2:]:l['bbox'][1]+=45.;l['bbox'][3]+=45.
        if mode=='caption':q['lines'][-2]['caption_start']=True
        if mode=='table':q['table_bboxes']=[[20.,400.,180.,461.]]
        if mode=='table-fragment':q['drawing_lines'].append(dict(bbox=[155.,460.,380.,460.],orientation='horizontal',width=.5));q['table_bboxes']=[[155.,420.,380.,465.]]
        if mode=='fraction':q['lines'].append(dict(line('numerator',[30.,450.,150.,458.],12),effective_height=8.));q['lines'][-3]['bbox']=[30.,462.,150.,470.]
        if mode=='visual-overlap':q['visual_bboxes']=[[20.,420.,180.,470.]]
        if mode=='short':q['drawing_lines'][0]['bbox'][2]=65.;q['lines'][-2]['footnote_marker_start']=True
        if mode=='tiny':q['drawing_lines'][0]['bbox'][2]=38.
        if mode in ['column','column-middle','column-regular','fullwidth']:q['drawing_lines'][0]['bbox'][2]=370.
        if mode=='column-middle':q['drawing_lines'][0]['bbox'][1]=q['drawing_lines'][0]['bbox'][3]=350.;q['lines'][-2]['bbox']=[30.,365.,250.,373.];q['lines'][-1]['bbox']=[30.,377.,250.,385.]
        if mode=='column-regular':
            for l in q['lines'][-2:]:l['effective_height']=12.;l['text']='Ordinary body text after section rule'
        if mode=='duplicate-rules':q['drawing_lines']*=2
        if mode=='lower-barrier':q['drawing_lines'].append(dict(bbox=[30.,480.,160.,480.],orientation='horizontal',width=.5))
        if mode=='edge-marker':q['lines'].append(dict(line('*',[20.,470.,26.,478.],12),effective_height=8.))
        if mode=='split-visual-row':q['lines'][-2]['text']='1.';q['lines'][-2]['bbox'][2]=38.;q['lines'].append(dict(line('Original native note begins here',[40.,470.,250.,478.],12),effective_height=8.))
        if mode=='restored-cluster':q['lines'][-2].update(restored_inline_cluster=True,bbox=[30.,470.,250.,486.])
        if mode=='style-height':q['lines'][-2].update(style_scale_repaired=True,em_height=8.,effective_height=18.)
        if mode=='footer-tail':q['lines'].append(dict(line('Footer',[30.,565.,100.,571.],12),effective_height=7.))
        if mode=='unicode-marker':q['lines'][-2]['text']='١. Original native note begins here'
        add(mode,q)
    for y in [329.9,330.,330.1,419.9,420.,420.1,479.9,480.,480.1]:
        for width in [23.9,24.,35.9,36.,47.9,48.,120.,221.,340.]:
            q=base();q['drawing_lines'][0]['bbox']=[30.,y,30.+width,y];q['lines'][-2]['bbox']=[30.,y+10.,250.,y+18.];q['lines'][-1]['bbox']=[30.,y+22.,250.,y+30.];add(f'rule/{y}/{width}',q)
    for mode in ['image','image-no-rule','image-frame','image-wide','image-offset','image-peer','image-big-note','image-gap','image-table','deferred-body']:
        q=base();q['visual_bboxes']=[[30.,300.,250.,450.]];q['drawing_lines'][0]['bbox']=[30.,455.,250.,455.]
        if mode=='image-no-rule':q['drawing_lines']=[]
        if mode=='image-frame':q['drawing_lines'][0]['bbox']=[30.,449.,250.,449.]
        if mode=='image-wide':q['drawing_lines'][0]['bbox'][2]=370.
        if mode=='image-offset':q['drawing_lines'][0]['bbox'][0]=100.
        if mode=='image-peer':q['visual_bboxes'].append([260.,300.,370.,455.])
        if mode=='image-big-note':
            for l in q['lines'][-2:]:l['effective_height']=12.
        if mode=='image-gap':q['lines'][-1]['bbox']=[30.,510.,250.,518.]
        if mode=='image-table':q['table_bboxes']=[[20.,410.,260.,456.]]
        if mode=='deferred-body':q['lines']=q['lines'][-2:];q['reference_body_height']=12.
        add(mode,q)
    for count in [1,2,3,4]:
        for mode in ['aligned','misaligned','numbered','gap','semantic','rotated']:
            q=base();q['lines']=[];q['drawing_lines']=[];q['table_bboxes']=[[20.,50.,380.,250.]]
            for r in range(count):
                for col in range(2):q['lines'].append(dict(line('中文字段'if mode!='numbered'else'一、字段',[30.+col*180.,300.+r*20.,100.+col*180.,310.+r*20.],r*2+col),effective_height=10.))
            if mode=='misaligned'and count>1:q['lines'][2]['bbox'][0]+=20.
            if mode=='gap'and count>1:q['lines'][2]['bbox'][1]+=30.;q['lines'][2]['bbox'][3]+=30.
            if mode=='semantic'and q['lines']:q['lines'][0]['semantic_type']='paragraph_title'
            if mode=='rotated'and q['lines']:q['lines'][0]['angle']=90
            add(f'parallel/{count}/{mode}',q)
    # Original upright->page rotation, including original-coordinate table barriers.
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    for angle in [90,180,270]:
        q=base()
        if angle in [90,270]:q['page_size']=q['page_size'][::-1]
        for l in q['lines']:l['bbox']=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),angle));l['angle']=angle
        for r in q['drawing_lines']:r['bbox']=list(_rotate_bbox_from_upright(tuple(r['bbox']),tuple(q['page_size']),angle))
        add('rotation/'+str(angle),q)
    return cases

def main():
    p=ROOT/'build/migration/native-early-footnotes-canonical.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=scenes();write_json(p,dict(cases=cases));print(len(cases),'old early footnote scenes;',sum(c['request']['lines']!=c['expected']['lines']for c in cases),'mutating scenes')
if __name__=='__main__':main()
