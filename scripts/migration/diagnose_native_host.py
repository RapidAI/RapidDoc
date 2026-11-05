"""Read-only host parity diagnostics; original and SDK run in separate processes."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import json,sys,hashlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
mode=sys.argv[1]
if mode=='original':
    import pypdfium2 as pdfium
    from rapid_doc.utils.pdf_text_tool import get_page
    def normalize(v):
        if isinstance(v,dict):return {k:normalize(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [normalize(x)for x in v]
        if hasattr(v,'bbox'):return normalize(v.bbox)
        if hasattr(v,'tolist'):return v.tolist()
        return v
else:
    import rapid_doc_rust as sdk
if mode=='original':from PIL import Image
out={}
for name,index,dpi in [('build/migration/model-document/five-pages.pdf',3,72),('demo/pdfs/比亚迪财报.pdf',2,144)]:
    data=(ROOT/name).read_bytes()
    if mode=='original':
        with pdfium.PdfDocument(data)as doc:
            p=doc[index];facts=normalize(get_page(p));im=p.render(scale=dpi/72).to_pil().convert('RGB')
    else:
        facts=sdk.read_native_pdf_page(data,page_index=index)
        image=sdk.render_pdf_page(data,page_index=index,dpi=dpi,color_mode='RGB')
        (ROOT/f'build/host-{mode}-{index}.png').write_bytes(image.png)
    if mode=='original':im.save(ROOT/f'build/host-{mode}-{index}.png')
    out[name]=dict(facts=facts)
(ROOT/f'build/host-diagnostic-{mode}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
