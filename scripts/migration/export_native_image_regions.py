"""Freeze old page-source raster decisions; image evidence remains an explicit stage input."""
import copy,json,random
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/native-image-regions.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import (_detect_repeated_raster_watermark_fingerprints as detect,_filter_repeated_raster_watermark_bboxes as filter_watermark,_exclude_proven_blank_top_under_display_title as trim)
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFImageInfo
    def page(images=(),lines=(),size=(400.,600.)):
        return dict(page_size=size,lines=lines,image_infos=images)
    def image(b=(0.,0.,400.,600.),fp=None,smooth=False,blank=None):
        return dict(bbox=b,fingerprint=fp,smooth_background=smooth,blank_top_bbox=blank)
    def line(text='body text',b=(20.,20.,100.,30.),h=10.,weight=None,angle=0):
        return dict(text=text,bbox=b,effective_height=h,dominant_font_weight=weight,angle=angle,source_index=0)
    def title_lines():
        return [line('a',h=10.),line('b',h=10.),line('Large title',(30.,10.,180.,18.),14.,600)]
    def title_image():return image((20.,10.,220.,200.),blank=(20.,20.,220.,200.))
    cases=[]
    def add(label,pages,pdf_pages=None):
        request=clean(dict(pages=copy.deepcopy(pages)))
        sources=[];infos=[];sizes=[]
        for p in copy.deepcopy(pages):
            rows=[]
            for raw in p['lines']:
                if raw.get('font_signature')is not None:raw['font_signature']=tuple(raw['font_signature'])
                raw['native_math_words']=frozenset(raw.get('native_math_words',[]))
                rows.append(_LineItem(**raw))
            sizes.append(tuple(p['page_size']))
            imgs=[PDFImageInfo(tuple(i['bbox']),i.get('fingerprint'),i.get('smooth_background',False),tuple(i['blank_top_bbox'])if i.get('blank_top_bbox')is not None else None)for i in p['image_infos']]
            infos.append(imgs);sources.append(SimpleNamespace(lines=rows,page_size=sizes[-1],image_bboxes=[]))
        watermarks=detect(infos,sizes)
        expected=[]
        for source,images in zip(sources,infos,strict=True):
            if len(source.lines)>=3 and sum(len(l.text.strip())for l in source.lines)>=60:images=[i for i in images if not i.smooth_background]
            source.image_bboxes=filter_watermark(images,source.page_size,watermarks)
            expected.append(trim(source,images))
        c=dict(label=label,request=request,expected=clean(dict(watermark_fingerprints=sorted(watermarks),image_bboxes=expected)))
        if pdf_pages is not None:c['pdf_pages']=pdf_pages
        cases.append(c)
    add('empty',[]);add('empty-page',[page()])
    for n in range(1,6):
        for fp in [None,'','repeat','other']:
            for copies in [1,3]:add(f'pages-{n}-{fp}-{copies}',[page([image(fp=fp)]*copies)for _ in range(n)])
    for ratio in [0.,.0799999,.08,.0800001,1.]:
        for n in [2,3,4]:
            for reverse in [False,True]:
                b=[0.,0.,400.,600.*ratio]
                if reverse:b=[b[2],b[3],b[0],b[1]]
                add(f'area-{ratio}-{n}-{reverse}',[page([image(b,'water'),image((0.,0.,10.,10.),'water')])for _ in range(n)])
    add('mixed-page-sizes',[page([image((0.,0.,100.,200.),'water')],size=s)for s in [(400.,600.),(100.,200.),(400.,600.),(0.,600.),(-10.,600.)]])
    add('different-fingerprints',[page([image(fp=str(i))])for i in range(5)])
    for rows in [2,3,4]:
        for chars in [19,20,21,30]:
            for text in ['x'*chars,'\t'+'x'*chars+'\u3000','😀'*chars,'\u200b'*chars]:
                add(f'body-{rows}-{chars}-{ord(text[0])}',[page([image(smooth=True),image((20.,20.,120.,120.),smooth=False)],[line(text)for _ in range(rows)])])
    # Statistics must include the background later discarded on the first two text-rich pages.
    add('detect-before-smooth',[page([image(fp='water',smooth=True)],[line('x'*20)]*3),page([image(fp='water',smooth=True)],[line('x'*20)]*3),page([image(fp='water')])])
    for field,values in [('height',[0.,10.,13.999999,14.,14.000001,30.]),('weight',[None,0,400,599,600,601]),('bottom',[11.,11.000001,18.,20.,20.000001]),('top',[19.999999,20.,21.]),('angle',[0,90,180,270]),('x',[[-60.,40.],[-59.999,40.001],[120.,320.],[120.001,320.001]])]:
        for value in values:
            rows=title_lines();r=rows[-1]
            if field=='height':r['effective_height']=value
            elif field=='weight':r['dominant_font_weight']=value
            elif field=='bottom':r['bbox']=list(r['bbox']);r['bbox'][3]=value
            elif field=='top':r['bbox']=list(r['bbox']);r['bbox'][1]=value
            elif field=='angle':r['angle']=value
            else:r['bbox']=[value[0],10.,value[1],18.]
            add(f'title-{field}-{value}',[page([title_image()],rows)])
    for heights in [[],[0.],[-1.],[10.,10.,10.],[10.,10.],[5.,15.,10.],[5.,15.,10.,10.],[10.,10.,100.]]:
        rows=[line(h=h)for h in heights]+[line('Title',(30.,10.,180.,18.),14.,600)]
        add('median-'+str(heights),[page([title_image()],rows)])
    add('no-upright-heights',[page([title_image()],[line(h=14.,angle=90,weight=600)])])
    for images in [[title_image(),image(title_image()['bbox'])],[image(title_image()['bbox']),title_image()],[title_image(),dict(image(title_image()['bbox'],smooth=True))]]:
        add('duplicate-'+str(len(cases)),[page(images,title_lines())])
    # Last bbox evidence wins, including when that duplicate is itself a discarded watermark.
    add('filtered-duplicate-proof',[page([title_image(),image(title_image()['bbox'],'w')],title_lines()),page([image(fp='w')]),page([image(fp='w')])])
    rng=random.Random(261028)
    for i in range(100):
        rows=title_lines();im=title_image();rows[-1]['bbox']=list(rows[-1]['bbox']);rows[-1]['bbox'][rng.randrange(4)]+=rng.choice([-15.,-.01,.01,15.]);rows[-1]['effective_height']=rng.choice([13.,14.,15.,20.]);add(f'geometry-{i}',[page([im],rows)])
    # Actual old visual/source-line inputs paired with actual old PDF image evidence.
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    corpora={name:json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))for name in ['native-visual-lines','native-visual-lines-extra']}
    grouped={}
    for i,c in enumerate(source['cases']):
        if 'visual_case'not in c:continue
        ref=c['visual_case'];v=corpora[ref['corpus']]['cases'][ref['index']]
        data=(ROOT/v['pdf']).read_bytes();assert digest(data)==v['pdf_sha256']
        with PDFDocument(data)as doc:images=[asdict(info)for info in doc.get_page_image_infos(v['page_index'])]
        p=page(images,c['expected']['lines'],c['request']['page_size'])
        pdf_ref=dict(source_case=i,pdf=v['pdf'],pdf_sha256=v['pdf_sha256'],page_index=v['page_index'])
        add('pdf-source-'+str(i),[p],[pdf_ref]);grouped.setdefault(v['pdf'],[]).append((v['page_index'],p,pdf_ref))
    for path,items in grouped.items():
        # Duplicate character-reader variants must not masquerade as different document pages.
        unique={idx:(p,ref)for idx,p,ref in items};items=[unique[idx]for idx in sorted(unique)]
        if len(items)>1:add('pdf-document-'+path,[p for p,_ in items],[ref for _,ref in items])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_objects.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_contracts.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'old image-region decisions;',sum('pdf_pages'in c for c in cases),'actual PDF source cases;',sum(bool(c['expected']['watermark_fingerprints'])for c in cases),'watermark-positive cases')
if __name__=='__main__':main()
