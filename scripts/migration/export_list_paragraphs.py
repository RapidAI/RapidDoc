"""冻结列表续行的作者几何合同、PDF 事实和独立旧 CPU OCR；不导入 Rust。"""
from io import BytesIO
import pypdfium2 as pdfium
from reportlab.pdfgen.canvas import Canvas
from common import ROOT, write_json, digest
from export_page_blocks import make_ocr
from pdf_corpus import extract_facts
from rapid_doc.utils.text_boundary import merge_text_line_contents


def main():
    def line(b, text):
        x0,y0,x1,y1=b
        return dict(points=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],text=text,confidence=None)
    def page(lines,label='text'):
        return dict(width=600,height=500,coordinate_space='page_pixels',blocks=[dict(
            layout_index=0,kind='text',region=dict(label=label,coordinate=[0,0,600,500]),lines=lines,text=None)])
    cases=[]
    def add(name,lines,groups,kinds=None,label='text'):
        cases.append(dict(name=name,page=page(lines,label),groups=groups,
                          kinds=kinds or ['list_item']*len(groups),texts=[
                              ('\n'.join(lines[i]['text']for i in group)if label=='algorithm'
                               else merge_text_line_contents([lines[i]['text']for i in group]))for group in groups]))
    lines=[line([10,10,240,30],'1. A long item'),line([35,40,240,60],'with a continuation'),
           line([36,70,160,90],'and a final line.'),line([10,100,240,120],'2. Next item'),
           line([35,130,240,150],'continued too'),line([10,160,240,180],'Ordinary body')]
    add('numbered-wrapped-body',lines,[[0,1,2],[3,4],[5]],['list_item','list_item','text'])
    for name,b,text in [
        ('flush-left',[10,40,240,60],'body'),
        ('excess-indent',[100,40,240,60],'far away'),
        ('vertical-gap',[35,65,240,85],'after a gap'),
        ('height-change',[35,40,240,67],'larger text'),
        ('new-marker',[35,40,240,60],'2. nested item'),
        ('blank-line',[35,40,240,60],''),
    ]:
        add(name,[line([10,10,240,30],'1. Item'),line(b,text)],[[0],[1]],
            ['list_item','list_item' if name=='new-marker' else 'text'])
    for marker in ['•','a)','(1)','一、','-']:
        add('marker-'+marker,[line([10,10,240,30],marker+' Long item'),
            line([35,40,240,60],'wrapped words')],[[0,1]])
    add('short-tail',[line([10,10,240,30],'• Item'),line([35,40,100,60],'end.'),
        line([35,70,240,90],'separate paragraph')],[[0,1],[2]],['list_item','text'])
    add('indent-drift',[line([10,10,240,30],'1. Item'),line([35,40,240,60],'continuation'),
        line([43,70,240,90],'new indent')],[[0,1],[2]],['list_item','text'])
    add('cjk',[line([10,10,240,30],'一、第一项'),line([35,40,240,60],'续行内容'),
        line([10,70,240,90],'二、第二项')],[[0,1],[2]])
    add('decimal-content',[line([10,10,240,30],'1. Values'),line([35,40,240,60],'3.14 is a value')],[[0,1]])
    add('two-columns',[line([10,10,240,30],'1. Left item'),line([330,10,560,30],'2. Right item'),
        line([35,40,240,60],'left continuation'),line([355,40,560,60],'right continuation')],[[0,2],[1,3]])
    add('title-is-not-list',[line([10,10,240,30],'1. Heading'),line([35,40,240,60],'continued heading')],
        [[0,1]],['title'],label='doc_title')
    add('algorithm-is-not-list',[line([10,10,240,30],'1. step'),line([35,40,240,60],'    next step')],
        [[0,1]],['algorithm'],label='algorithm')

    output=ROOT/'build/migration/list-paragraphs';output.mkdir(parents=True,exist_ok=True)
    contents=['1. First item contains several words','with another line of useful detail',
              '2. Second item contains several words','with another line of useful detail',
              'Ordinary text starts at the left margin','and continues as a separate paragraph']
    buf=BytesIO();canvas=Canvas(buf,pagesize=(600,320),invariant=1);canvas.setFont('Helvetica',24)
    for i,text in enumerate(contents):canvas.drawString(88 if i in (1,3) else 60,270-32*i,text)
    canvas.save();data=buf.getvalue();pdf=output/'list.pdf';pdf.write_bytes(data)
    with pdfium.PdfDocument(data)as doc:
        source=doc[0];bitmap=source.render(scale=1)
        image=bitmap.to_pil().convert('RGB');bitmap.close();source.close()
    png=output/'list.png';image.save(png)
    ocr=make_ocr()(image)
    assert len(ocr)==6,ocr
    write_json(ROOT/'build/migration/list-paragraphs.json',dict(cases=cases,runtime=dict(
        pdf_input=str(pdf),pdf_sha256=digest(data),png_input=str(png),rgb_sha256=digest(image.tobytes()),
        width=image.width,height=image.height,ocr=ocr,facts=extract_facts(data),
        source_lines=contents,groups=[[0,1],[2,3],[4,5]],kinds=['list_item','list_item','text'],
        expected_ocr=[merge_text_line_contents([ocr[i]['text']for i in group])for group in [[0,1],[2,3],[4,5]]],
        expected_native=[merge_text_line_contents(contents[i:i+2])for i in (0,2,4)])))
    print(f'Exported {len(cases)} list continuation contracts and independent 6-line OCR/native PDF fixture')


if __name__=='__main__':main()
