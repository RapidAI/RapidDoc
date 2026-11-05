"""Freeze original classifier decisions/edge rules; separate original Python process."""
import importlib.util
from io import BytesIO
import itertools
import json
from pathlib import Path
from common import ROOT,digest,write_json

def helper(filename):
    spec=importlib.util.spec_from_file_location('reference_'+filename,ROOT/'tests'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    target=ROOT/'build/migration/native-classification-rules.json'
    docs=ROOT/'build/migration/native-classification-documents.json'
    if target.exists()or docs.exists():raise FileExistsError('Classification references already exist')
    from rapid_doc.utils import pdf_classify as old
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import RectangleObject
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    sparse=helper('test_pdf_classify_sparse_text.py');cid=helper('test_pdf_classify_cid_quality.py')
    corpus=[];folder=ROOT/'build/migration/native-classification-pdfs';folder.mkdir(exist_ok=True)
    def add(label,data):
        path=folder/(label+'.pdf')
        if path.exists():raise FileExistsError(path)
        path.write_bytes(data);expected=old.classify(data)
        corpus.append(dict(label=label,source=path.relative_to(ROOT).as_posix(),sha256=digest(data),expected=expected))
    for names in itertools.product(['blank','sparse','scan','dense'],repeat=2):add('-'.join(names),sparse.pdf([getattr(sparse,name)for name in names]))
    for value in ['','FFFF','0000','0001']:add('cid-'+(value or 'readable'),cid.cid_pdf(value))
    for kind,rotation,cropped in itertools.product(['background','covered'],[0,90,180,270],[False,True]):
        stream=BytesIO();canvas=Canvas(stream,pagesize=(300,400));image=Image.new('RGB',(300,400),(225,235,245))
        if kind=='background':canvas.drawImage(ImageReader(image),0,0,300,400)
        for line in range(8):canvas.drawString(20,360-line*25,'Normal native text stays readable.')
        if kind=='covered':canvas.drawImage(ImageReader(image),0,0,300,400)
        canvas.save();image.close();reader=PdfReader(BytesIO(stream.getvalue()));page=reader.pages[0];page.rotate(rotation)
        if cropped:page.cropbox=RectangleObject([10,20,290,380])
        writer=PdfWriter();writer.add_page(page);out=BytesIO();writer.write(out);add(f'{kind}-{rotation}-crop{int(cropped)}',out.getvalue())
    requests=[]
    def case(op,request,expected):requests.append(dict(operation=op,request=request,expected=expected))
    for count in [0,1,2,9,10,11,13,100,100001]:
        for maximum in [0,1,2,10]:case('indices',dict(count=count,max=maximum),old.get_sample_page_indices(count,maximum))
    texts=['','Plain words 123','中'*100+'犮'*30,'中文'*200+'А'*50+'ا'*50+'क'*50,'A'*80+'.'*100,'A'*79+'.'*100,'A'*75+'!?;:'*25,'\0\ufffd\ue000\x1f中']
    samples=[]
    for index,text in enumerate(texts):
        s=dict(page_index=index,text=text,cleaned_text=''.join(c for c in text if not c.isspace()),char_count=len(text),null_char_count=text.count('\0'),replacement_char_count=text.count('\ufffd'),control_char_count=sum(old._is_disallowed_control_unicode(ord(c))for c in text),private_use_char_count=sum(0xe000<=ord(c)<=0xf8ff for c in text),unicode_map_error_count=0)
        samples.append(s)
    for group in [[s]for s in samples]+[samples,samples[::-1]]:
        for op,function in [('quality',old._get_text_quality_signal_from_samples),('map_errors',old._get_unicode_map_error_signal_from_samples),('u72',old._get_u72xx_text_signal_from_samples),('cross',old._get_cross_script_text_signal_from_samples),('punctuation',old._get_sampled_ascii_punct_signal_from_samples)]:case(op,dict(samples=group),function(group))
    for count,error,bad in itertools.product([29,30,100],[0,3,4],[0,2,3]):
        sample=cid.sample({'ReadableCID':count,'Other':4900},errors={'ReadableCID':error},abnormal={'ReadableCID':bad})
        fonts={'triggered':True,'page_fonts':{0:['ReadableCID']}}
        case('cid',dict(samples=[sample],fonts=fonts),old._get_cid_font_usage_signal_from_samples([sample],fonts))
        sample['font_non_generated_cjk_char_counts']={'ReadableCID':int(count*.8)}
        case('latin',dict(samples=[sample],fonts=fonts),old._get_latin_font_cjk_usage_signal_from_samples([sample],fonts,30,.01,.8))
    sources={p:digest((ROOT/p).read_bytes())for p in ['rapid_doc/utils/pdf_classify.py','rapid_doc/utils/pdf_image_text_probe.py']}
    write_json(target,dict(sources=sources,cases=requests));write_json(docs,dict(sources=sources,cases=corpus))
    print('Original classifier references:',len(requests),'rule cases,',len(corpus),'actual PDF documents')
if __name__=='__main__':main()
