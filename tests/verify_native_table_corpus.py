"""Compare changed edge handling against its previous implementation in memory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inspect
import json
import fitz
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page,recover_table_region
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector

root=Path(__file__).resolve().parents[1]
out=root/'output-table-01-diagnostic'
fixed_collapse=vector._collapse_outer_duplicate_tracks
fixed_build=vector._build_vector_tracks
source=inspect.getsource(fixed_collapse)
start=source.index('            # A generous snap tolerance')
end=source.index('            combined =',start)
exec(source[:start]+source[end:],vector.__dict__)
baseline_collapse=vector._collapse_outer_duplicate_tracks
exec(inspect.getsource(fixed_build).replace('collapse_leading_edge=True','collapse_leading_edge=False'),vector.__dict__)
baseline_build=vector._build_vector_tracks

def recover(evidence,bbox,baseline):
    vector._collapse_outer_duplicate_tracks=baseline_collapse if baseline else fixed_collapse
    vector._build_vector_tracks=baseline_build if baseline else fixed_build
    return recover_table_region(evidence,bbox)

records=[]
page_count=0
files=['05.pdf','demo1.pdf','GBT3620.1-2016.pdf','49－长表格-密集型.pdf',
       '比亚迪财报.pdf','示例1-论文模板.pdf','示例7-研究报告.pdf']
for filename in files:
    path=root/'demo/pdfs'/filename
    with fitz.open(path) as visual, PDFDocument(path.read_bytes()) as document:
        for index in range(min(10,len(visual))):
            page_count+=1
            tables=visual[index].find_tables().tables
            if not tables:
                continue
            evidence=prepare_table_page(document[index])
            for number,table in enumerate(tables):
                for delta in (0,-.4,.4):
                    bbox=tuple(v+delta for v in table.bbox)
                    before=recover(evidence,bbox,True)
                    after=recover(evidence,bbox,False)
                    old=bool(before and before.confidence>=.9)
                    new=bool(after and after.confidence>=.9)
                    records.append({'file':filename,'page':index+1,'table':number+1,'offset':delta,
                        'before':old,'after':new,'html_changed':bool(old and new and before.html!=after.html)})
    print(filename,'done',flush=True)
vector._collapse_outer_duplicate_tracks=fixed_collapse
vector._build_vector_tracks=fixed_build
summary={'files':len(files),'pages':page_count,'regions_including_offset_probes':len(records),
    'before_accepted':sum(r['before'] for r in records),'after_accepted':sum(r['after'] for r in records),
    'lost':sum(r['before'] and not r['after'] for r in records),
    'previously_accepted_html_changes':sum(r['html_changed'] for r in records)}
(out/'cross-document-validation.json').write_text(json.dumps({'summary':summary,'regions':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(summary)
