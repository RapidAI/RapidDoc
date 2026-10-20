"""Replay native layout with old/new continuation hints, without changing files."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inspect
import json
import fitz
from loguru import logger
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules

logger.remove()
root=Path(__file__).resolve().parents[1]
out=root/'output-table-continuation-fixed'
out.mkdir(exist_ok=True)
fixed=table_rules._trailing_table_continuation_grid
source=inspect.getsource(fixed)
start=source.index('    # Pagination continues')
end=source.index('    if len(positions)',start)
oldsource=source[:start]+'    positions = _closed_grid_vertical_track_positions(horizontal, axis_lines, em)\n'+source[end:]
exec(oldsource,table_rules.__dict__)
baseline=table_rules._trailing_table_continuation_grid
table_rules._trailing_table_continuation_grid=fixed
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
                data=subset.tobytes()
                subset.close()
        table_rules._trailing_table_continuation_grid=baseline
        before=analyze_docvortex_layout(data)
        table_rules._trailing_table_continuation_grid=fixed
        after=analyze_docvortex_layout(data)
        changed=[i+1 for i,(a,b) in enumerate(zip(before,after)) if a!=b]
        def tables(pages):
            return sum(b['type']=='table' for p in pages for b in p)
        records.append({'file':filename,'pages':len(after),'before_tables':tables(before),
                        'after_tables':tables(after),'changed_pages':changed})
        if filename=='01.pdf':
            (out/'01_native_layout.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
            assert len([b for b in after[26] if b['type']=='table'])==1
            table=next(b for b in after[26] if b['type']=='table')
            (out/'page27-table.html').write_text(table['content'],encoding='utf-8')
        print(records[-1],flush=True)
finally:
    table_rules._trailing_table_continuation_grid=fixed
(out/'cross-document-validation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
