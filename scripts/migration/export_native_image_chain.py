"""Complete old graphic/raster builders: representative branch scenes, no candidate injection."""
import copy,json
from common import ROOT,digest,write_json
from export_native_page_containers import typed
from export_native_inline_clusters import clean
from export_native_table_page_policies import empty
from export_native_bar_groups import base as bars

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as g
    s=typed(q['source']);fields=['semantic_type','title_suppressed','paragraph_group','median_glyph_width'];before=[[getattr(l,k)for k in fields]for l in s.lines]
    if q['kind']=='graphic':blocks,claimed=g._build_graphic_like_blocks(s,[tuple(b)for b in q.get('excluded_bboxes',[])],set(q.get('claimed_line_indices',[])))
    else:blocks,claimed=g._build_raster_image_blocks(s,q.get('container_blocks',[]),set(q.get('claimed_line_indices',[])))
    raw=copy.deepcopy(q['source'])
    for r,l,prior in zip(raw['lines'],s.lines,before):
        for key,previous in zip(fields,prior):
            if getattr(l,key)!=previous:r[key]=clean(getattr(l,key))
    return clean(dict(source=raw,claimed_line_indices=sorted(claimed),**{'graphic_blocks'if q['kind']=='graphic'else'raster_image_blocks':blocks}))

def line(text,b,i=0,**kw):return dict(text=text,bbox=b,source_index=i,angle=0,effective_height=10.,font_signature=None,chars=[],**kw)
def main():
    target=ROOT/'build/migration/native-image-chain.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,s,kind='raster',**kw):
        for l in s['lines']:l.setdefault('angle',0);l.setdefault('chars',[])
        q=dict(source=copy.deepcopy(s),kind=kind,**kw);cases.append(dict(label=label,request=q,expected=reference(q)))
    for h in [True,False]:
        for compound in [False,True]:add(f'bars/{h}/{compound}',bars(h,compound),'graphic')
    for mode in ['basic','empty','claimed','blocked','full-background','photo-shadow','photo-border-text','signature','vertical-tiles','inline-three','inline-wrong-row','inline-extra-body']:
        s=empty();s['page_size']=[400.,600.];s['image_bboxes']=[[40.,100.,160.,200.]];s['lines']=[line('Original native label',[60.,120.,140.,130.])];kw={}
        if mode=='empty':s['lines']=[]
        if mode=='claimed':kw['claimed_line_indices']=[0]
        if mode=='blocked':kw['container_blocks']=[dict(type='table',bbox=[20.,80.,200.,250.])]
        if mode=='full-background':s['image_bboxes']=[[0.,0.,400.,600.]];s['lines']=[line('These original native body lines must remain independent from the background',[20.,50.+i*20,350.,60.+i*20],i)for i in range(9)]
        if mode in ['photo-shadow','photo-border-text']:
            s['image_bboxes'].append([42.,102.,158.,198.])
            if mode=='photo-border-text':s['lines'].append(line('Edge label',[40.,99.,42.,102.],1))
        if mode=='signature':s['signature_bboxes']=[[10.,10.,14.,12.],[10.2,10.,14.,12.]]
        if mode=='vertical-tiles':s['image_bboxes']=[[120.,100.+i*11,200.,110.+i*11]for i in range(3)];s['lines']=[]
        if mode.startswith('inline-'):
            s['image_bboxes']=[[20.+i*26,200.,40.+i*26,260.]for i in range(3)];s['lines']=[line('+',[40.+i*26,210.,46.+i*26,220.],i,split_from_row=True,visual_row_id=99 if mode!='inline-wrong-row'else i)for i in range(2)]
            if mode=='inline-extra-body':s['lines'].append(line('body',[40.,242.,46.,252.],2))
        add(mode,s,**kw)
    # Complete graphic stages retain old fallback/empty results as well as positives.
    s=bars();s['lines']+= [line('Fig 1: A descriptive title',[50.,70.,180.,80.],200),line('continued title wording',[50.,82.,180.,92.],201)];add('wrapped-caption',s,'graphic')
    s=bars();s['lines'].append(line('Long original body line ending next to the graphic',[20.,62.,380.,72.],200));s['lines'].append(line('short tail',[20.,74.,80.,84.],201));add('body-tail',s,'graphic')
    refs={str(p.relative_to(ROOT)):digest(p.read_bytes())for p in [ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/isolated_graphics.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_merging.py']};write_json(target,dict(sources=refs,cases=cases));print(len(cases),'complete graphic/raster branches',sum(bool(c['expected'].get('graphic_blocks')or c['expected'].get('raster_image_blocks'))for c in cases),'positive scenes')
if __name__=='__main__':main()
