"""冻结实际PDF旧DocVortex原语→旧矢量候选，以及轨道/原子格硬门补充。"""
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_table_vector_grid import plain
def main():
    target=ROOT/'build/migration/native-table-vector-pdf.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableText,NativeTableRectangle
    cases=[];real=[]
    def add(t,text):
        d=[];result=old.build_vector_candidates(t,text,d)
        request=asdict(t);request['chars']=[]
        cases.append(dict(request=dict(table_input=request,text=asdict(text)),expected=plain(dict(candidates=[asdict(c)for c in result],diagnostics=d,html=[serialize_candidate_html(c)for c in result]))))
    for filename,page,bbox in [('build/migration/model-document/five-pages.pdf',2,(18.,38.,222.,162.)),('tests/migration/inputs/pdf_native.pdf',0,(20.,200.,180.,320.))]:
        path=ROOT/filename
        with PDFDocument(path.read_bytes())as doc:
            p=prepare_table_page(doc[page])
            for angle in [0,90,180,270]:
                t=NativeTableInput(bbox,p.page_size,angle,tuple(p.geometry.chars),p.drawing_lines,p.rectangles)
                text=build_native_table_text(t)
                if text is None:continue
                # 请求不携带原Bbox字符对象；旧字形事实独立冻结，Rust不读取这个PDF。
                from dataclasses import replace
                t=replace(t,chars=())
                i=len(cases);add(t,text);real.append(dict(pdf=filename,pdf_sha256=digest(path.read_bytes()),page_index=page,angle=angle,case_index=i))
    for rows,cols in [(2,200),(101,101)]:
        lines=tuple([NativeTableRule((0.,r*30.-.1,cols*50.,r*30.+.1),.2,'horizontal')for r in range(rows+1)]+[NativeTableRule((c*50.-.1,0.,c*50.+.1,rows*30.),.2,'vertical')for c in range(cols+1)])
        t=NativeTableInput((0.,0.,cols*50.,rows*30.),(10000.,10000.),0,(),lines,())
        add(t,NativeTableText((),(),10.,1.))
    rect=NativeTableRectangle((10.,10.,20.,20.),5,False,True)
    add(NativeTableInput((0.,0.,150.,90.),(200.,200.),0,(),(),(rect,)*1251),NativeTableText((),(),10.,10.))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'+p for p in ['vector.py','text.py']]+['rapid_doc/_vendor/docvortex_core/analyzers/pdf.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,real=real))
    print(f'Frozen {len(cases)} additional vector cases including {len(real)} actual PDF/angle inputs')
if __name__=='__main__':main()
