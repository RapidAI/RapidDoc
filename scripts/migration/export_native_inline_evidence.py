"""Freeze real old inline detection functions, including source-stage PDF chains."""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/native-inline-evidence.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_AxisLine
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.detection import detect_pdf_text_style_lines,detect_pdf_text_link_lines
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFLinkAnnotation
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    cases=[]
    def add(label,lines,drawings=(),annotations=(),chain=None):
        objects=[_LineItem(**copy.deepcopy(l))for l in lines]
        rules=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in drawings]
        links=[PDFLinkAnnotation(a['target'],tuple(tuple(b)for b in a['bboxes']),a.get('source_index',i))for i,a in enumerate(annotations)]
        expected=clean(dict(style_lines=[asdict(s)for s in detect_pdf_text_style_lines(objects,rules)],link_lines=[asdict(s)for s in detect_pdf_text_link_lines(objects,links)]))
        c=dict(label=label,request=clean(dict(lines=lines,drawing_lines=drawings,annotations=annotations)),expected=expected)
        if chain is not None:c['source_case']=chain
        cases.append(c)
    def line(text='abcdef',font='Helvetica',flags=0,weight=400,y=20.,source=0):
        chars=[dict(char=c,char_idx=i,bbox=[10.+i*6.,y,16.+i*6.,y+10.],font=dict(name=font,flags=flags,weight=weight))for i,c in enumerate(text)]
        return dict(text=text,bbox=[10.,y,10.+len(text)*6.,y+10.],angle=0,source_index=source,chars=chars)
    def rule(y,x0=10.,x1=46.,width=1.,orientation='horizontal'):
        return dict(bbox=[x0,y-.05,x1,y+.05],width=width,orientation=orientation)
    def annotation(target='https://example.test/a',boxes=([10.,20.,28.,30.],)):
        return dict(target=target,bboxes=boxes,source_index=0)
    for name in ['Helvetica','ABCDEF+Test-Bold','Demi','Black','Heavy','Foo-bd','foo-biMT','FOOgbi','Foo-w6','Foo-w5','foo-İ','Foo-Bold\n','Foo-Bold\nX']:
        for flags in [0,64,1<<18]:add(f'font-{name}-{flags}',[line(font=name,flags=flags)])
    for weight in [0,599.9,600,900,'600',True,None]:add('weight-'+str(weight),[line(weight=weight)])
    for text in ['a','ab','ﬁ','ﬃ','•abc','•• abc','• abc','\x02a','a\u00adb','A\u200bB','a\t b','\U0001f600x','', '  ']:
        l=line(text,font='Test-Bold')
        if text.startswith('•'):
            for c in l['chars'][text.count('•'):]:c['font']['name']='Helvetica'
        add('compact-'+repr(text),[l])
    for y in [24.9,25.,27.,28.,30.,32.,32.01]:
        for width in [0.,2.,2.01]:add(f'drawing-{y}-{width}',[line()],[rule(y,width=width)])
    for x0,x1 in [(10.,27.9),(10.,28.),(10.,46.),(-10.,70.),(20.,40.),(46.,10.),(10.,10.)]:add(f'extent-{x0}-{x1}',[line()],[rule(30.,x0,x1)])
    for top in [30.,31.,31.5,31.51,35.]:add('fraction-'+str(top),[line(source=3),line(y=top,source=1)],[rule(30.)])
    for ink in [[10.,27.,46.,29.],[10.,24.,46.,26.],None]:
        l=line()
        for c in l['chars']:c['tight_bbox']=ink
        add('ink-'+str(ink),[l],[rule(25.)])
    for index in [0,1,2]:
        l=line();l['chars'][index]['font']['flags']=1<<18;add('short-bold-'+str(index),[l])
    for angle in [0,90,180,270,360,45]:
        l=line(font='Test-Bold');l['angle']=angle;add('rotation-'+str(angle),[l],[rule(30.)],[annotation()])
    for kind in ['reverse','duplicate','missing','float','bool']:
        l=line(font='Test-Bold');l['chars'].reverse()
        for c in l['chars']:
            if kind=='duplicate':c['char_idx']=1
            elif kind=='missing':c.pop('char_idx')
            elif kind=='float':c['char_idx']=float(c['char_idx'])
            elif kind=='bool':c['char_idx']=bool(c['char_idx']%2)
        add('order-'+kind,[l],[rule(30.)],[annotation()])
    for text in ['a bﬁ','ﬁﬃ','ab\u00adcd','A\x02B','😀x',' a\t\u00a0b']:
        for boxes in [([10.,20.,28.,30.],),([10.,20.,16.,30.],[28.,20.,46.,30.]),([11.,21.,15.,29.],)]:add('link-text-'+repr(text)+'-'+str(boxes),[line(text)],annotations=[annotation(boxes=boxes)])
    for link_target in ['', 'https://a.test','https://b.test']:
        add('link-conflict-'+link_target,[line()],annotations=[annotation(),annotation(link_target)])
    for offset in [-.01,0.,.01,3.,6.]:add('link-center-'+str(offset),[line()],annotations=[annotation(boxes=([10.+offset,20.,13.+offset,30.],))])
    add('mixed-page',[line(font='Bold',source=9),line(y=60.,source=3)],annotations=[annotation()])
    add('tie-drawing',[line(source=9),line(source=2)],[rule(30.)])
    rng=random.Random(27001)
    for i in range(80):
        l=line(rng.choice(['a ﬁ b','one two','123456','• abc']));l['source_index']=rng.randint(-2,10)
        for c in l['chars']:c['font']['weight']=rng.choice([400,600,700])
        add('random-'+str(i),[l],[rule(rng.choice([25.,30.,40.]),width=rng.choice([.5,1.,3.]))],[annotation(),annotation('https://conflict.test',([22.,20.,34.,30.],))])
    source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    vc={name:json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))for name in ['native-visual-lines','native-visual-lines-extra']}
    for i,c in enumerate(source['cases']):
        if'visual_case'not in c:continue
        r=c['visual_case'];v=vc[r['corpus']]['cases'][r['index']]
        with PDFDocument((ROOT/v['pdf']).read_bytes())as doc:links=[asdict(a)for a in doc.get_page_link_annotations(v['page_index'])]
        add('PDF-source-'+str(i),c['expected']['lines'],c['expected']['drawing_lines'],links,chain=i)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/common.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'cases;',sum(bool(c['expected']['style_lines'])for c in cases),'style pages;',sum(bool(c['expected']['link_lines'])for c in cases),'link pages;',sum('source_case'in c for c in cases),'PDF chains')
if __name__=='__main__':main()
