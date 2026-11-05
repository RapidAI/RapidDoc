"""Focused real Form tree + full old normalize/materialize chain; immutable sources."""
import copy,json
from dataclasses import asdict
from io import BytesIO
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_page_containers import typed as partial_typed,project_maps,page_reference
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA

def typed(raw):
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import _PDFFormInfo
    s=partial_typed(raw);infos=[]
    for f in raw.get('form_infos',[]):
        f=copy.deepcopy(f)
        for k in ['bbox','declared_bbox','media_bbox','matrix','clip','occurrence']:
            if f[k]is not None:f[k]=tuple(f[k])
        for k in ['text_indices','path_indices']:f[k]=frozenset(f[k])
        f['image_bboxes']=tuple(tuple(b)for b in f['image_bboxes']);infos.append(_PDFFormInfo(**f))
    s.form_infos=tuple(infos)
    s.retained_page_forms={tuple(b)for b in raw.get('retained_page_forms',[])}
    return s

def normalized_reference(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.form_roles import normalize_page_forms
    sources=[typed(s)for s in raw];before=[copy.deepcopy(s)for s in raw];normalize_page_forms(sources);out=[]
    for original,s in zip(before,sources):
        r=project_maps(original,s)
        if s.form_infos:
            r['form_bboxes']=clean(s.form_bboxes);r['retained_page_forms']=sorted(clean(s.retained_page_forms));r['form_member_sources']=[dict(bbox=clean(b),source_indices=sorted(ids))for b,ids in sorted(s.form_member_sources.items())];r['form_path_sources']=[dict(bbox=clean(b),source_indices=sorted(ids))for b,ids in sorted(s.form_path_sources.items())]
            for raw_path,p in zip(r['path_infos'],s.path_infos):raw_path['form_depth']=p.form_depth
            for l,real in zip(r['lines'],s.lines):
                for k in ['semantic_type','paragraph_group','title_suppressed','median_glyph_width']:
                    old=l.get(k,False if k=='title_suppressed'else None);value=getattr(real,k)
                    if old!=value:l[k]=clean(value)
            if 'drawing_component_cache'in r:r['drawing_component_cache']=[]
        out.append(r)
    return clean(out)

def document_reference(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _build_form_image_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    from types import SimpleNamespace as NS
    result=[]
    for source in normalized_reference(raw):
        out=page_reference(dict(source=source));s=typed(out['source']);protected={l.source_index for l in typed(source).lines if l.semantic_type is not None};protected|={l.source_index for l in s.lines if l.semantic_type in ['index','paragraph_title']}
        # Actual page stage uses the analysis members chosen before table commit.
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.index_blocks import _extract_index_blocks
        before=typed(source);_extract_index_blocks(before.lines,before.page_size,before.image_bboxes+before.signature_bboxes,require_heading=True,require_explicit_heading=True);protected={l.source_index for l in before.lines if l.semantic_type is not None};analysis=__import__('dataclasses').replace(s,lines=[l for l in s.lines if l.source_index not in protected]);blocks,claims=_build_form_image_blocks(analysis,[tuple(b)for b in out['active_form_bboxes']],set(out['claimed_line_indices']))
        for r,l in zip(out['source']['lines'],s.lines):
            if r.get('median_glyph_width')!=l.median_glyph_width:r['median_glyph_width']=l.median_glyph_width
        claimed=set(out['claimed_line_indices'])|claims;out['claimed_line_indices']=sorted(claimed);out['remaining_lines']=[r for r in out['source']['lines']if r['source_index']not in claimed];out['form_image_blocks']=clean(blocks);out['fixed_blocks']=out['rule_code_blocks']+out['annotation_blocks']+out['table_blocks']+out['code_blocks']+clean(blocks)+out['index_blocks'];grid=_trailing_table_continuation_grid(s,NS(fixed_blocks=out['fixed_blocks'],remaining_lines=[l for l in s.lines if l.source_index not in claimed]));out['next_table_grid']=None if grid is None else clean(dict(page_index=grid.page_index,angle=grid.angle,positions=grid.positions,tolerance=grid.tolerance));result.append(out)
    return clean(dict(pages=result))

def main():
    target=ROOT/'build/migration/native-form-runtime.json';pdf=ROOT/'build/migration/form-runtime.pdf'
    if target.exists()or pdf.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from loguru import logger
    logger.remove();buf=BytesIO();c=Canvas(buf,pagesize=(400,600),pageCompression=1)
    c.beginForm('inner',0,0,180,180);c.setFont('Helvetica',10);c.drawString(15,130,'Inner label');c.rect(10,10,160,150);c.line(10,10,160,150);c.endForm()
    c.beginForm('wrapper',0,0,400,600);c.saveState();c.translate(90,200);c.doForm('inner');c.restoreState();c.setFont('Helvetica',10)
    for i in range(8):c.drawString(30,560-i*18,'Long original body line with enough native words and width')
    c.endForm()
    c.beginForm('sparse',0,0,400,600);c.setFont('Helvetica',10)
    for i in range(3):c.drawString(30,560-i*18,'Long original body line with enough native words and width')
    c.endForm()
    c.beginForm('underlay',0,0,400,600);c.rect(0,0,400,600,fill=1,stroke=0);c.endForm()
    c.beginForm('clipped',0,0,180,180);p=c.beginPath();p.rect(30,30,100,100);c.clipPath(p,stroke=0,fill=0);c.setFont('Helvetica',10);c.drawString(10,110,'Clipped original line');c.rect(0,0,180,180);c.doForm('inner');c.endForm()
    labels=[]
    for name,rot in [('inner',0),('inner',90),('inner',180),('inner',270),('wrapper',0),('sparse',0),('underlay',0),('clipped',0),('repeat',0),('precision',0)]:
        c.setPageRotation(rot)
        if name=='repeat':
            for x in [20,210]:c.saveState();c.translate(x,200);c.doForm('inner');c.restoreState()
        elif name=='precision':c.saveState();c.transform(1.000123456,0.000123456,-0.000987654,1.000234567,31.123456789,80.987654321);c.doForm('inner');c.restoreState()
        else:c.doForm(name)
        if name=='underlay':c.setFont('Helvetica',10);c.drawString(20,300,'External body above decorative underlay')
        c.showPage();labels.append(name+'/'+str(rot))
    c.save();pdf.write_bytes(buf.getvalue());data=pdf.read_bytes();cases=[];sources=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            with doc._open_page(i)as page:frame=list(page.get_bbox())
            fields=list(dict.fromkeys(FIELDS+EXTRA+['median_glyph_width','ink_bbox','run_index']));source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in fields}for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes,image_bboxes=[list(info.bbox)for info in snap.image_infos],signature_bboxes=[],page_index=i));sources.append(source)
            expected=clean(dict(page_size=snap.page_size,page_bbox=frame,rotation=snap.rotation,form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes));cases.append(dict(label=label,page_index=i,expected=expected))
    refs={str(p.relative_to(ROOT)):digest(p.read_bytes())for p in [ROOT/'rapid_doc/_vendor/docvortex_core/document/pdf/form_structure.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/form_roles.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py']}
    write_json(target,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),sources=refs,geometry=cases,request=dict(sources=sources),normalized=normalized_reference(sources),expected=document_reference(sources)));print('Real Form pages',len(cases),'Form instances',sum(len(c['expected']['form_infos'])for c in cases))

if __name__=='__main__':main()
