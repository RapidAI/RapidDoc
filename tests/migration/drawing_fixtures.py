"""Source pages and middle facts for vector-overlay compatibility checks."""
from copy import deepcopy
from io import BytesIO
from pypdf import PdfReader,PdfWriter
from pypdf.generic import NameObject,NumberObject,RectangleObject,DictionaryObject,DecodedStreamObject
from reportlab.pdfgen import canvas


def source(rotation=0,crop=False):
    buffer=BytesIO();c=canvas.Canvas(buffer,pagesize=(400,320))
    c.setFillColorRGB(.9,.95,1);c.rect(10,10,360,280,fill=1,stroke=0)
    c.setFillColorRGB(0,0,0);c.drawString(20,290,'Original vector content 123')
    c.line(10,10,380,300);c.save()
    writer=PdfWriter();page=writer.add_page(PdfReader(BytesIO(buffer.getvalue())).pages[0])
    page[NameObject('/Rotate')]=NumberObject(rotation)
    if crop:page[NameObject('/CropBox')]=RectangleObject([11,17,390,310])
    # Existing names must remain valid when the new overlay has its own fonts.
    form=DecodedStreamObject();form.set_data(b'0 0 1 rg 0 0 8 8 re f')
    form.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Form'),NameObject('/BBox'):RectangleObject([0,0,10,10])})
    page['/Resources'][NameObject('/XObject')]=DictionaryObject({NameObject('/RD_Overlay0'):writer._add_object(form)})
    page['/Resources']['/Font'][NameObject('/RD_Font')]=page['/Resources']['/Font']['/F1']
    out=BytesIO();writer.write(out);return out.getvalue()


def facts():
    def line(box,kind='text',index=0,**extra):
        return {'bbox':box,'index':index,'spans':[dict(type=kind,bbox=box,content='text',**extra)]}
    def block(kind,box,index=0,**extra):
        return dict(type=kind,bbox=box,lines=[line(box,index=index)],**extra)
    table=block('table',[20,150,250,230],blocks=[
        block('table_body',[20,170,250,210],virtual_lines=[]),
        block('table_caption',[20,150,250,165]),
        block('table_footnote',[20,212,250,226]),
        block('table_footnote',[20,230,250,240],cross_page=True),
    ])
    table['blocks'][0]['lines']=[line([22,172,240,205],'table',7,img_boxes=[[30,180,50,195]],latex_boxes=[{'bbox':[55,180,90,195],'polygon_points':[[55,181],[90,180],[90,195],[55,195]]}])]
    image=block('image',[180,20,260,110],blocks=[
        block('image_body',[180,20,260,70],virtual_lines=[]),block('image_caption',[180,72,260,85]),block('image_footnote',[180,87,260,100])])
    image['blocks'][0]['lines']=[line([180,20,260,70],'image',8)]
    blocks=[
        block('title',[20,20,150,40],2),
        block('text',[20,50,150,75],4,polygon_points=[[20,50],[150,48],[145,76],[20,75]]),
        block('ref_text',[20,80,150,95],1),
        block('interline_equation',[20,100,150,115],6),
        block('list',[20,120,150,145],5,blocks=[block('text',[25,125,145,140])]),
        block('index',[270,20,350,40],3),image,table,
        block('code',[270,150,350,210],blocks=[block('code_body',[270,160,350,210]),block('code_caption',[270,145,350,158])]),
    ]
    for b in blocks:
        if b['type']=='interline_equation':b['lines'][0]['spans'][0]['type']='interline_equation'
    blocks[1]['lines'][0]['spans'] += [dict(type='inline_equation',bbox=[110,52,125,70]),dict(type='checkbox',bbox=[130,53,140,65]),dict(type='text',bbox=[20,50,80,75],polygon_points=[[20,50],[80,49],[80,75],[20,75]])]
    return dict(page_idx=0,page_size=[400,320],para_blocks=blocks,preproc_blocks=deepcopy(blocks),discarded_blocks=[block('discarded',[290,260,340,280])])
