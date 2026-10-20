"""Compare local flow ordering against the prior title-only correction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import fitz
from loguru import logger
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis import native_boundaries

logger.remove()
root=Path(__file__).resolve().parents[1]
out=root/'output-leading-flow-fixed'
out.mkdir(exist_ok=True)
fixed=native_boundaries._order_leading_shared_flow
records=[]
try:
    for filename in ['01.pdf','05.pdf','demo1.pdf','GBT3620.1-2016.pdf','49－长表格-密集型.pdf',
                     '比亚迪财报.pdf','示例1-论文模板.pdf','示例7-研究报告.pdf']:
        with fitz.open(root/'demo/pdfs'/filename) as doc:
            if filename=='01.pdf':
                data=doc.tobytes()
            else:
                subset=fitz.open()
                subset.insert_pdf(doc,from_page=0,to_page=min(9,len(doc)-1))
                data=subset.tobytes();subset.close()
        native_boundaries._order_leading_shared_flow=lambda blocks,size:list(blocks)
        before=analyze_docvortex_layout(data)
        native_boundaries._order_leading_shared_flow=fixed
        after=analyze_docvortex_layout(data)
        changed=[i+1 for i,(a,b) in enumerate(zip(before,after)) if a!=b]
        # Reordering must not change block type, text, geometry or multiplicity.
        key=lambda b:json.dumps(b,ensure_ascii=False,sort_keys=True)
        assert all(sorted(a,key=key)==sorted(b,key=key) for a,b in zip(before,after))
        records.append({'file':filename,'pages':len(after),'changed_pages':changed,'block_contents_unchanged':True})
        if filename=='01.pdf':
            (out/'01_native_layout.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
            assert '第三部分' in str(after[0][0]['content'])
            assert '1.库内' in str(after[0][1]['content'])
        print(records[-1],flush=True)
finally:
    native_boundaries._order_leading_shared_flow=fixed
(out/'cross-document-validation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
