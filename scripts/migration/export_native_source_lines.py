"""Freeze actual old page-source line preparation and its complete side effects."""
import copy,json,random
from dataclasses import fields,asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/native-source-lines.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_text import restore_drop_cap_paragraphs,group_hanging_letter_paragraphs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _extract_decorative_text_rules,_coerce_pdf_drawing_lines
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDrawingLine
    names=[f.name for f in fields(_LineItem)if not f.name.startswith('_')]
    cases=[]
    def add(label,raw,page=(400.,600.),images=(),rules=(),chain=None):
        request=clean(dict(lines=raw,page_size=page,image_infos=images,drawing_lines=rules))
        objects=[]
        for r in copy.deepcopy(raw):
            if r.get('font_signature')is not None:r['font_signature']=tuple(r['font_signature'])
            r['native_math_words']=frozenset(r.get('native_math_words',[]));objects.append(_LineItem(**r))
        restore_drop_cap_paragraphs(objects);drop=sum(l.paragraph_group is not None for l in objects)
        group_hanging_letter_paragraphs(objects);grouped=sum(l.paragraph_group is not None for l in objects)
        drawing=[PDFDrawingLine((0.,0.),(0.,0.),tuple(r['bbox']),r['width'],r['orientation'])for r in rules]
        drawing=_coerce_pdf_drawing_lines(drawing)
        heights=sorted(l.em_height for l in objects if l.em_height>0)
        if heights:
            em=heights[len(heights)//2]
            raster=[PDFDrawingLine((b[0],(b[1]+b[3])/2),(b[2],(b[1]+b[3])/2),tuple(b),b[3]-b[1],'horizontal')for info in images for b in [info['bbox']]if b[2]-b[0]>=2*em and 0<b[3]-b[1]<=.1*em]
            drawing.extend(_coerce_pdf_drawing_lines(raster))
        kept,decoration=_extract_decorative_text_rules(objects,tuple(page));drawing.extend(decoration)
        c=dict(label=label,request=request,expected=clean(dict(lines=[{k:getattr(l,k)for k in names}for l in kept],drawing_lines=[asdict(r)for r in drawing])),events=dict(drop=drop,grouped=grouped,decorative=len(decoration),raster=len(drawing)-len(decoration)-len(rules)))
        if chain:c['visual_case']=chain
        cases.append(c)
    def line(text,bbox,index,**kw):return dict(text=text,bbox=bbox,angle=0,source_index=index,effective_height=10.,em_height=10.,font_signature=['Body',0],**kw)
    def char(text,bbox):return dict(char=text,bbox=bbox)
    def drop():
        first=line('A  quick brown fox',[10.,10.,110.,50.],0)
        first['chars']=[char('A',[10.,10.,24.,50.]),char(' ',[24.,20.,24.,20.])]+[char(c,[25.+i*5.,20.,30.+i*5.,30.])for i,c in enumerate('quick')]+[char(' ',[50.,20.,51.,30.])]
        peers=[line('wrapped text',[25.,y,110.,y+10.],i+1)for i,y in enumerate([32.,44.])]
        peers+=[line('returned body',[10.,y,110.,y+10.],i+3)for i,y in enumerate([56.,68.])]
        return [first,*peers]
    def hanging():
        return [line('a)',[10.,10.,18.,20.],0),line('one two three four five',[28.,10.,150.,20.],1),line('continuation words',[28.,24.,150.,34.],2),line('the last line',[28.,38.,150.,48.],3)]
    add('drop-positive',drop());add('hanging-positive',hanging())
    for first in [True,False]:
        for h in [0.,5.,8.49,8.5,10.,11.5,11.51,20.]:
            rows=drop()if first else hanging();rows[2]['effective_height']=h;add(f'height-{first}-{h}',rows)
        for semantic in ['title','text',None]:
            rows=drop()if first else hanging();rows[2]['semantic_type']=semantic;add(f'barrier-{first}-{semantic}',rows)
        for text in ['1) next','a. next','１２) next','a)\tnext','a)next','A) next','3. next']:
            rows=drop()if first else hanging();rows[2]['text']=text;add(f'number-{first}-{text}',rows)
        for group in [-2,0,8]:
            rows=drop()if first else hanging();rows[-1]['paragraph_group']=group;add(f'existing-group-{first}-{group}',rows)
        for angle in [0,90,180,270]:
            rows=drop()if first else hanging();rows[0]['angle']=angle;add(f'angle-{first}-{angle}',rows)
    for extra in ['host','fragment','foreign-tail']:
        rows=hanging()
        if extra=='host':rows.append(line('other one two three four',[30.,10.,155.,20.],8))
        elif extra=='fragment':rows.append(line('x',[75.,24.,80.,34.],8))
        else:rows.append(dict(line('italic tail',[28.,52.,150.,62.],8),font_signature=['Italic',0]))
        add('hanging-'+extra,rows)
    for n in [1,2,3,11,12]:
        rows=drop();rows[0]['chars']=rows[0]['chars'][:2]+[char('a',[25.+i*5.,20.,30.+i*5.,30.])for i in range(n)]+[char(' ',[90.,20.,91.,30.])];add('suffix-'+str(n),rows)
    for spacing in [0.,1.]:
        rows=drop();rows[0]['chars'][1]['bbox'][2]+=spacing;add('space-geometry-'+str(spacing),rows)
    for marker in ['a)','z.','A)','ab)','1)','a)\n',' a) ']:
        rows=hanging();rows[0]['text']=marker;add('marker-'+marker,rows)
    for em in [[10.],[5.,15.],[10.,0.,20.],[]]:
        rows=[line('body',[10.,40.+i*15.,100.,50.+i*15.],i)for i in range(len(em))]
        for r,h in zip(rows,em):r['em_height']=h
        images=[dict(bbox=[0.,20.,width,20.+h])for width in [19.,20.,30.,40.]for h in [0.,.5,1.,1.01,2.]]
        add('raster-em-'+str(em),rows,images=images,rules=[dict(bbox=[20.,40.,10.,40.],width=-1.,orientation='horizontal')])
    for count in [7,8,10,12]:
        for text in ['\ue000'*count,'\ue000'*(count-2)+'xx','x'*count,'\U000f0000'*count,'\U00100000'*count]:
            for angle in [0,90,180,270]:
                # Rotated bbox computed with the same source geometry helper, not manually changed expected output.
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                b=_rotate_bbox_from_upright((10.,20.,210.,25.),(400.,600.),angle)
                add(f'decoration-{count}-{ord(text[0])}-{angle}',[dict(line(text,b,0),angle=angle)])
    rng=random.Random(261027)
    for i in range(80):
        rows=drop()if i%2 else hanging()
        row=rng.choice(rows);axis=rng.choice([0,1,2,3]);row['bbox'][axis]+=rng.choice([-6.,-.1,.1,6.]);add('geometry-'+str(i),rows)
    for name in ['native-visual-lines','native-visual-lines-extra']:
        corpus=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for index,c in enumerate(corpus['cases']):
            if'pdf'not in c:continue
            add(f'visual-chain-{name}-{index}',c['expected'],c['request']['page_size'],chain=dict(corpus=name,index=index))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/local_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'source-line cases;',sum(c['events']['drop']>0 for c in cases),'drop,',sum(c['events']['grouped']>c['events']['drop']for c in cases),'hanging,',sum(c['events']['decorative']for c in cases),'decorative,',sum(c['events']['raster']for c in cases),'raster,',sum('visual_case'in c for c in cases),'PDF chains')
if __name__=='__main__':main()
