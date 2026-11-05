"""Freeze unchanged Python PdfModel on the two user PDF samples; never load the SDK."""
import json,time
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_table_media_png import projection

def main():
    target=ROOT/'build/migration/native-pdf-acceptance.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel
    cases=[]
    for name in ['示例1-论文模板.pdf','比亚迪财报.pdf']:
        path=ROOT/'demo/pdfs'/name;data=path.read_bytes();start=time.perf_counter()
        with PDFDocument(data)as doc:
            pages=PdfModel().predict(doc)
        cases.append(dict(label=name,pdf=path.relative_to(ROOT).as_posix(),pdf_sha256=digest(data),expected=clean(projection(pages))))
        print(name,len(pages),'pages',round(time.perf_counter()-start,2),'seconds',flush=True)
    write_json(target,dict(cases=cases))
    predictions=ROOT/'build/migration/native-pdf-predictions.json'
    if predictions.exists():raise FileExistsError(predictions)
    from rapid_doc._vendor.docvortex_core.content import normalize_pdf_model_text
    f=json.loads((ROOT/'build/migration/native-pdf-document.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        normalize_pdf_model_text(c['expected']['pages'])
        c.pop('collected')
    write_json(predictions,f)

if __name__=='__main__':main()
