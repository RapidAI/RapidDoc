"""Fresh full-state real PDF reference and actual positive local-metric documents."""
import copy,json
from common import ROOT,write_json,digest
from export_native_geometry_plans import page,rotate,reference
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
def local(bad=6,donor_count=2,donor_chars=16):
    p=page(rows=bad,n=16,height=40.,font=1.,spacing=18.,name='BodyRegular');d=page(rows=donor_count,n=donor_chars,height=12.8,font=1.,spacing=20.,name='BodyBold')
    # Assign new coordinates: the source factory intentionally aliases raw/loose boxes.
    for l in d['source']['lines']:
        l['source_index']+=bad;l['bbox']=[v+300. if i%2 else v for i,v in enumerate(l['bbox'])]
        for c in l['chars']:c['char_idx']+=1000;c['bbox']=[v+300. if i%2 else v for i,v in enumerate(c['bbox'])]
    for key in ['tight_bboxes','loose_bboxes','origins']:
        field='origin'if key=='origins'else'bbox'
        for r in d['geometry'][key]:r['char_idx']+=1000;r[field]=[v+300. if i%2 else v for i,v in enumerate(r[field])]
        p['geometry'][key]+=d['geometry'][key]
    p['source']['lines']+=d['source']['lines'];p['source']['chars']=[copy.deepcopy(c)for l in p['source']['lines']for c in l['chars']];p['geometry']['chars']=copy.deepcopy(p['source']['chars']);return p
def main():
    real_target=ROOT/'build/migration/native-geometry-real-document.json';target=ROOT/'build/migration/native-geometry-plan-proofs.json'
    if real_target.exists()or target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,pages):
        q=dict(pages=copy.deepcopy(pages))
        for p in q['pages']:p['source']['lines']=[project(l)for l in lines(p['source']['lines'])]
        cases.append(dict(label=label,request=q,expected=reference(q)))
    for bad in [5,6,8]:
        for donors in [0,1,2]:add(f'local/{bad}/{donors}',[local(bad,donors)])
    add('local/insufficient-donor-width',[local(donor_chars=10)])
    for angle in [90,180,270]:add(f'local/rotation/{angle}',[rotate(local(),angle),page(rows=6,height=40.,spacing=14.)])
    positive=sum(bool(c['expected']['plan']['local_metric_repair_sources'])for c in cases);assert positive>=3,positive
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    original=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text(encoding='utf-8'));pdf=ROOT/original['pdf'];pages=[]
    with PDFDocument(pdf.read_bytes())as doc:
        for i in range(len(original['request']['sources'])):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            rows=lambda m,k:[dict(char_idx=i,**{k:list(v)})for i,v in sorted(m.items())]
            geometry=dict(chars=clean(g.chars),tight_bboxes=rows(g.tight_bboxes,'bbox'),origins=rows(g.origins,'origin'),loose_bboxes=rows(g.loose_bboxes,'bbox'));pages.append(dict(source=dict(lines=[project(l)for l in ls],page_size=list(snap.page_size)),geometry=geometry))
    q=dict(pages=pages);result=reference(q);write_json(real_target,dict(pdf=original['pdf'],pdf_sha256=digest(pdf.read_bytes()),case=dict(label='actual-full-state-document',request=q,expected=result)));write_json(target,dict(cases=cases));print(len(cases),'local-metric documents;',positive,'local positives;',len(pages),'real full-state PDF pages')
if __name__=='__main__':main()
