"""Freeze original raster-connected diagram and plain-text frame rules."""
import copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-raster-diagrams.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_raster_connected_diagrams as detect,_is_plain_text_frame as plain
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_AxisLine
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    def rule(b,d):return dict(bbox=b,width=1.,orientation=d)
    def line(b):return dict(text='cell',bbox=b,angle=0,source_index=0)
    def source():return dict(page_size=[400.,600.],drawing_lines=[rule([20.,y,220.,y],'horizontal')for y in [50.,100.,150.,200.]]+[rule([x,50.,x,200.],'vertical')for x in [20.,70.,120.,170.,220.]]+[rule([40.,y,200.,y],'horizontal')for y in [80.,130.,180.]],image_bboxes=[[30.,60.,50.,75.],[140.,60.,160.,75.],[30.,160.,50.,175.],[140.,160.,160.,175.]],path_infos=[],lines=[],form_bboxes=[])
    def textframe():return dict(page_size=[400.,600.],drawing_lines=[rule([20.,50.,220.,50.],'horizontal'),rule([20.,200.,220.,200.],'horizontal'),rule([20.,50.,20.,200.],'vertical'),rule([220.,50.,220.,200.],'vertical')],image_bboxes=[],path_infos=[],lines=[line([40.,70.,100.,80.])],form_bboxes=[])
    cases=[]
    def add(label,s,frames=(),em=10.):
        q=clean(dict(source=copy.deepcopy(s),frame_bboxes=frames,em=em));old=SimpleNamespace(**dict(s,drawing_lines=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in s['drawing_lines']],lines=[_LineItem(**l)for l in copy.deepcopy(s['lines'])],path_infos=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox'])))for p in s['path_infos']],drawing_component_cache=[]))
        expected=clean(dict(raster_diagram_bboxes=detect(old,em),plain_text_frame_flags=[plain(old,tuple(b),em)for b in frames]));cases.append(dict(label=label,request=q,expected=expected))
    box=[20.,50.,220.,200.];add('diagram-positive',source(),[box]);add('plain-positive',textframe(),[box]);add('empty',dict(page_size=[400.,600.],drawing_lines=[],image_bboxes=[],path_infos=[],lines=[],form_bboxes=[]),[box])
    for count in [0,3,4,5]:
        s=source();s['image_bboxes']=(s['image_bboxes']*2)[:count];add('image-count-'+str(count),s,[box])
    for count in [0,11,12]:
        s=source();s['drawing_lines']=s['drawing_lines'][:count];add('rule-count-'+str(count),s)
    for n in [2,3]:
        s=source();s['drawing_lines']=s['drawing_lines'][:9]+s['drawing_lines'][9:9+n];add('interior-count-'+str(n),s)
    for x in [29.9999,30.,30.0001]:
        s=source()
        for r in s['drawing_lines'][9:]:r['bbox'][0]=x
        add('interior-left-'+str(x),s)
    for x in [209.9999,210.,210.0001]:
        s=source()
        for r in s['drawing_lines'][9:]:r['bbox'][2]=x
        add('interior-right-'+str(x),s)
    for x in [18.9999,19.,19.0001]:
        s=source();s['image_bboxes'][0]=[x,60.,x+20.,75.];add('image-overlap-'+str(x),s)
    for em in [0.,1.,10.,24.9999,25.,25.0001,30.]:add('em-'+str(em),source(),[box],em)
    s=source();s['lines']=[line([x,y,x+30.,y+10.])for x in [30.,140.]for y in [60.,110.,160.]];add('table-protected',s)
    for form in [box,[300.,300.,350.,350.]]:
        s=source();s['lines']=[line([x,y,x+30.,y+10.])for x in [30.,140.]for y in [60.,110.,160.]];s['form_bboxes']=[form];add('form-'+str(form),s)
    for shift in [0.,1.,2.,2.0001]:add('frame-overlap-'+str(shift),textframe(),[[20.+shift,50.,220.+shift,200.]])
    for n in [0,1]:
        s=textframe();s['lines']=s['lines'][:n];add('frame-text-'+str(n),s,[box])
    for image in [[40.,70.,60.,90.],[0.,0.,400.,600.]]:
        s=textframe();s['image_bboxes']=[image];add('frame-image-'+str(image),s,[box])
    for segments in [8,9]:
        s=textframe();s['path_infos']=[dict(bbox=[40.,70.,80.,100.],segment_count=segments,fill_visible=False,stroke_visible=True,form_depth=0,source_index=0)];add('frame-complex-'+str(segments),s,[box])
    for count in [3,4,5]:
        s=textframe();s['drawing_lines']=(s['drawing_lines']*2)[:count];add('frame-rules-'+str(count),s,[box])
    rng=random.Random(261106)
    for i in range(100):
        s=source()if i%2 else textframe();b=rng.choice(s['drawing_lines'])['bbox'];b[rng.randrange(4)]+=rng.choice([-.001,.001,-8.,8.]);add('geometry-'+str(i),s,[box])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'old raster diagram/frame facts;',sum(bool(c['expected']['raster_diagram_bboxes'])for c in cases),'diagram-positive;',sum(any(c['expected']['plain_text_frame_flags'])for c in cases),'plain-frame-positive')
if __name__=='__main__':main()
