"""Freeze original auto classification/statistics for the reported and existing PDFs."""
import json
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/api-classification.json'
    if target.exists():raise FileExistsError(target)
    import pypdfium2 as pdfium
    from rapid_doc.utils.pdf_classify import classify,get_sample_page_indices,_collect_pdfium_text_samples
    names=['demo1.pdf','49－长表格-密集型.pdf','03.pdf','small_ocr.pdf','示例1-论文模板.pdf','比亚迪财报.pdf']
    cases=[]
    for name in names:
        path=ROOT/'demo/pdfs'/name;data=path.read_bytes()
        with pdfium.PdfDocument(data)as doc:
            indexes=get_sample_page_indices(len(doc));samples=_collect_pdfium_text_samples(doc,indexes)
        cases.append(dict(source=path.relative_to(ROOT).as_posix(),sha256=digest(data),expected=classify(data),samples=samples))
        print(name,cases[-1]['expected'],len(samples),'sampled pages',flush=True)
    write_json(target,dict(cases=cases))
if __name__=='__main__':main()
