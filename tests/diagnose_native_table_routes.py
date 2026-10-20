"""Read-only replay of saved table regions against the native recovery engine."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import fitz
from collections import Counter
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page,recover_table_region
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.engine import diagnose_native_pdf_table
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput

root=Path(__file__).resolve().parents[1]
out=root/'output-table-01-diagnostic'
out.mkdir(exist_ok=True)
saved=json.loads((root/'output1/01/auto/01_model.json').read_text(encoding='utf-8'))
records=[]
visual_document=fitz.open(root/'demo/pdfs/01.pdf')
with PDFDocument((root/'demo/pdfs/01.pdf').read_bytes()) as document:
    for idx,page in enumerate(saved):
        tables=[b for b in page['layout_dets'] if b.get('category_id')==5]
        if not tables:
            continue
        evidence=prepare_table_page(document[idx])
        for j,table in enumerate(tables):
            poly=table['poly']
            box=tuple(poly[k]/(200/72) for k in [0,1,4,5])
            result=recover_table_region(evidence,box)
            record={'page':idx+1,'table':j+1,'bbox':box,'characters':len(evidence.geometry.chars),
                    'lines':len(evidence.drawing_lines),'rectangles':len(evidence.rectangles),
                    'confidence':result.confidence if result else None,
                    'source':result.source if result else None,
                    'accepted':bool(result and result.html.strip() and result.confidence>=0.9)}
            if result:
                record['diagnostics']=result.diagnostics
                (out/f'p{idx+1}-t{j+1}.html').write_text(result.html,encoding='utf-8')
            if not record['accepted']:
                inp=NativeTableInput(table_bbox=box,page_size=evidence.page_size,angle=0,
                    chars=tuple(evidence.geometry.chars),drawing_lines=evidence.drawing_lines,rectangles=evidence.rectangles)
                record['failure_diagnostics']=diagnose_native_pdf_table(inp)
                candidates=visual_document[idx].find_tables().tables
                def iou(candidate):
                    a=fitz.Rect(box); b=fitz.Rect(candidate.bbox)
                    intersection=(a & b).get_area()
                    return intersection/(a.get_area()+b.get_area()-intersection)
                closest=max(candidates,key=iou,default=None)
                if closest is not None and iou(closest)>0.9:
                    snapped=recover_table_region(evidence,closest.bbox)
                    record['physical_bbox_probe']={'bbox':closest.bbox,'iou':iou(closest),
                        'confidence':snapped.confidence if snapped else None,
                        'accepted':bool(snapped and snapped.confidence>=0.9)}
            records.append(record)
        print('page',idx+1,'tables',len(tables),'accepted',sum(r['accepted'] for r in records if r['page']==idx+1),flush=True)
summary={'tables':len(records),'accepted':sum(r['accepted'] for r in records),
         'sources':dict(Counter(r['source'] for r in records)),
         'failed':[(r['page'],r['table']) for r in records if not r['accepted']]}
summary['physical_bbox_probe_recovered']=sum(r.get('physical_bbox_probe',{}).get('accepted',False) for r in records)
visual_document.close()
(out/'native-replay.json').write_text(json.dumps({'summary':summary,'tables':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(summary)
