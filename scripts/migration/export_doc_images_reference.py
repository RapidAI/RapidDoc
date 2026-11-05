from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
import olefile
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.fib import parse_fib
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.formatting import parse_formatting_runs
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.records import DocBudget
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.images import ImageStore,inline_picture

from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.doc_converter import DocConverter
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.sprm import chpx_picture_location
with olefile.OleFileIO('demo/ms_office_docs/docx_01.doc') as o:
 w=o.openstream('WordDocument').read();f=parse_fib(w);t=o.openstream('1Table' if f.base.uses_1table else '0Table').read();d=o.openstream('Data').read();a,b=f.pair(12),f.pair(13);budget=DocBudget()
 runs=parse_formatting_runs(w,t,d,chpx_offset=a.fc,chpx_size=a.lcb,papx_offset=b.fc,papx_size=b.lcb,budget=budget)
 offsets=sorted({v for r in runs.characters if (v:=chpx_picture_location(r.grpprl)) is not None});store=ImageStore();rows=[]
 for offset in offsets:
  v=inline_picture(d,offset=offset,store=store,budget=budget);rows.append(dict(offset=offset,exists=v is not None and (bool(v.equation_latex) or bool(DocConverter._serialize_image(v))),equation=v.equation_latex if v else None))
Path('tests/fixtures/office_rust_reference_068/doc_images.json').write_text(json.dumps(rows),encoding='utf-8');print('PICF references',len(rows),'formulas',sum(bool(v['equation'])for v in rows))
