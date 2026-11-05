"""Freeze old captioned concentric contour candidates without altered predicates."""
import copy,json,random,math
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-concentric-graphics.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_captioned_concentric_path_graphics as detect
    def path(b,seg=8,stroke=True):return dict(bbox=b,segment_count=seg,stroke_visible=stroke,fill_visible=False,form_depth=0,source_index=0)
    def line(text,b,angle=0):return dict(text=text,bbox=b,angle=angle,source_index=0)
    def source():return dict(page_size=[400.,600.],path_infos=[path([80.,100.,180.,200.]),path([100.,120.,160.,180.])],lines=[line('Figure 1. caption',[80.,70.,180.,80.]),line('25%',[85.,130.,110.,140.]),line('75 %',[145.,160.,175.,170.])])
    cases=[]
    def add(label,s,em=10.):
        request=clean(dict(source=copy.deepcopy(s),em=em));old=SimpleNamespace(page_size=tuple(s['page_size']),lines=[_LineItem(**l)for l in copy.deepcopy(s['lines'])],path_infos=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox'])))for p in s['path_infos']]);cases.append(dict(label=label,request=request,expected=clean(dict(concentric_bboxes=detect(old,em)))))
    add('positive',source());add('empty',dict(page_size=[400.,600.],path_infos=[],lines=[]))
    for text in ['Figure 1','FIG.2','fig3abc','FigURE\t１２','fİg ١','fıg 1','Fig\u00a01','图1','Figure one','figu1','figures 1','figure. 1',' figure 1','fig\n1','Figure 1x: caption','Figure 1K: title','fig1.2','fig','']:
        s=source();s['lines'][0]['text']=text;add('caption-'+text,s)
    for text in ['25%',' １２ %　','١٢,٥%','+25%','-25%','25. %','25.%','25.5 %','25,5%','%','25','25％','25% text','\t25\n%\r','²%','1e2%']:
        s=source();s['lines'][1]['text']=text;add('percent-'+text,s)
    for angle in [0,90,180,270,15]:
        for index in [0,1]:
            s=source();s['lines'][index]['angle']=angle;add(f'angle-{index}-{angle}',s)
    for index in [0,1]:
        for field,values in [('form_depth',[-1,0,1]),('segment_count',[6,7,8]),('stroke_visible',[False,True])]:
            for value in values:
                s=source();s['path_infos'][index][field]=value;add(f'path-{index}-{field}-{value}',s)
    for shift in [7.9999,8.,8.0001,-8.,-8.0001]:
        s=source();b=s['path_infos'][1]['bbox'];b[0]+=shift;b[2]+=shift;add('center-'+str(shift),s)
    for width in [39.999,40.,40.001,69.999,70.,130.,130.001,160.,160.001]:
        s=source();s['path_infos'][0]['bbox'][2]=80.+width;add('width-'+str(width),s)
    for bottom in [0.,-.0001,99.9999,100.,100.0001,200.]:
        s=source();s['lines'][0]['bbox'][3]=bottom;add('caption-gap-'+str(bottom),s)
    for left in [-20.,-19.999,130.,130.001]:
        s=source();s['lines'][0]['bbox']=[left,70.,left+100.,80.];add('caption-overlap-'+str(left),s)
    for gap in [49.999,50.,50.001]:
        s=source();s['lines'][1]['bbox']=[180.+gap,130.,200.+gap,140.];add('percentage-gap-'+str(gap),s)
    for seg in [1,2,5,6]:
        for width in [69.999,70.,70.001]:
            s=source();s['path_infos'].append(path([70.,120.,70.+width,150.],seg));add(f'leader-{seg}-{width}',s)
    for size in [[249.999,600.],[250.,600.],[250.001,600.],[400.,249.999],[400.,250.],[0.,600.]]:
        s=source();s['page_size']=size;add('size-'+str(size),s)
    for em in [0.,5.,10.,15.,15.0001]:add('em-'+str(em),source(),em)
    s=source();s['path_infos'].append(copy.deepcopy(s['path_infos'][0]));add('duplicate-source-id-and-box',s)
    s=source();s['path_infos'].reverse();add('reverse-paths',s)
    s=source();s['path_infos'][0]['bbox']=[80,100,180,200];s['path_infos'][1]['bbox']=[100.,120,160,180.];add('mixed-types',s)
    rng=random.Random(261105)
    for i in range(80):
        s=source();b=rng.choice(s['path_infos'])['bbox'];b[rng.randrange(4)]+=rng.choice([-.001,.001,-8.,8.]);add('geometry-'+str(i),s)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'old concentric cases;',sum(bool(c['expected']['concentric_bboxes'])for c in cases),'positive')
if __name__=='__main__':main()
