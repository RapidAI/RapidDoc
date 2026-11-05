"""Freeze original PPTX package geometry, group filtering and XYCut++ order.

This is a development reference generator, never a runtime dependency.
"""
import base64
import gzip
import json
import random
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.oxml.ns import qn
    from lxml import etree
    from PIL import Image
    from loguru import logger
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.pptx_converter import PptxConverter
    from rapid_doc._vendor.docvortex_core.analyzers.native._shared.xycut import sort_entries

    logger.remove()
    cases = []
    def paragraph_text(engine, p, shape):
        return engine._build_paragraph_rich_text(p,shape)

    def add(name, raw):
        case = dict(name=name, package_b64=base64.b64encode(raw).decode())
        try:
            try:
                prs = Presentation(BytesIO(raw))
            except Exception as direct_error:
                from rapid_doc._vendor.docvortex_core.analyzers.native.office.pptx.package_normalizer import normalize_pptx_package
                normalized = normalize_pptx_package(raw)
                if normalized == raw: raise
                prs = Presentation(BytesIO(normalized))
                # Geometry checkpoint receives normalized input. This does not
                # claim that Rust's still-pending retry normalizer is tested.
                case.update(reference_normalized_input=True, original_error=type(direct_error).__name__,
                    package_b64=base64.b64encode(normalized).decode())
            engine = PptxConverter()
            # Accessing .slides renames slide parts in python-pptx. Preserve
            # actual OPC names before that lazy side effect for native lookup.
            part_names={id(p):str(p.partname).lstrip('/')for p in prs.part.package.iter_parts()}
            width, height = int(prs.slide_width or 9144000), int(prs.slide_height or 6858000)
            slides = []
            for slide in prs.slides:
                flat = engine._flatten_slide_shapes(slide.shapes)
                shapes = []
                for i, entry in enumerate(flat):
                    is_picture = engine._safe_shape_type(entry.shape) == MSO_SHAPE_TYPE.PICTURE
                    paragraphs=[]
                    if entry.shape.has_text_frame:
                        for p in entry.shape.text_frame.paragraphs:
                            sources=engine._get_paragraph_font_sources(entry.shape,p)
                            paragraphs.append(dict(**engine._build_paragraph_style_profile(entry.shape,p),
                                runs=[dict(styles=(engine._get_style_str_from_run(r,sources)or'').split(',') if engine._get_style_str_from_run(r,sources) else [],
                                    font_size_pt=engine._resolve_effective_run_font_size_pt(r,sources))for r in p.runs],
                                list=engine._get_paragraph_list_info(entry.shape,p)))
                    texts=[paragraph_text(engine,p,entry.shape)for p in entry.shape.text_frame.paragraphs] if entry.shape.has_text_frame else []
                    blocks=[]
                    if any(t is None for t in texts):blocks=None
                    elif entry.shape.has_text_frame:
                        engine.cur_page=[];engine._handle_text_elements(entry.shape)
                        engine._cleanup_slide_text_block_metadata(engine.cur_page)
                        blocks=engine.cur_page
                    collected=None if blocks is None else engine._collect_shape_blocks(entry,flat,i,width,height)
                    if collected is not None:engine._cleanup_slide_text_block_metadata(collected)
                    shapes.append(dict(id=entry.shape.shape_id, bbox=entry.bbox,
                        has_text=engine._shape_has_raw_text(entry.shape), picture=is_picture,
                        paragraphs=paragraphs,
                        text=texts,blocks=blocks,collected=collected,
                        skip_picture=is_picture and engine._should_skip_picture(entry, flat[i+1:], width, height)))
                def related(part, kind):
                    return next((r.target_part for r in part.rels.values() if not r.is_external and r.reltype.endswith('/'+kind)), None)
                layout = related(slide.part, 'slideLayout')
                master = related(layout, 'slideMaster') if layout else None
                notes = related(slide.part, 'notesSlide')
                name_of = lambda part: part_names.get(id(part),str(part.partname).lstrip('/')) if part else None
                slides.append(dict(part=name_of(slide.part), layout=name_of(layout), master=name_of(master),
                    notes=name_of(notes), shapes=shapes,
                    order=[b['index'] for b in sort_entries([dict(bbox=e.bbox, index=i)for i,e in enumerate(flat)], beta=2., density_threshold=.9)]))
            case['expected'] = dict(width=width, height=height, slides=slides)
        except Exception as exc:
            case['error'] = type(exc).__name__
        cases.append(case)

    paths = [ROOT/'demo/pptx/powerpoint_sample.pptx', ROOT/'tests/fixtures/docvortex_migration/pptx_01.pptx',
             ROOT/'tests/migration/inputs/synthetic.pptx', *sorted((ROOT/'tests/fixtures/pptx').glob('*.pptx'))]
    for path in paths:
        if path.exists(): add(path.relative_to(ROOT).as_posix(), path.read_bytes())

    def saved(prs):
        stream=BytesIO();prs.save(stream);return stream.getvalue()
    for case_id in range(18):
        deck=Presentation();s=deck.slides.add_slide(deck.slide_layouts[6]);table=s.shapes.add_table(3,4,100,100,6000000,3000000).table
        for r in range(3):
            for c in range(4):
                cell=table.cell(r,c);cell.text='row '+str(r)+' col '+str(c)+' <&> "'
                run=cell.text_frame.paragraphs[0].runs[0];run.font.bold=case_id%2==0;run.font.italic=case_id%3==0;run.font.underline=case_id%4==0
                if case_id%3==0:run.hyperlink.address='https://example.com/?x=1&y=2'
                if case_id%2:cell.text_frame.add_paragraph().text='next\vline'
        if case_id%3==0:table.cell(0,0).merge(table.cell(1,1))
        if case_id%3==1:table.cell(1,1).merge(table.cell(2,3))
        if case_id>=12:
            table.cell(0,0)._tc.set('rowSpan',['0','-1','20','bad','2',''][case_id-12])
        add('resources/table/'+str(case_id),saved(deck))
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    for kind in [XL_CHART_TYPE.COLUMN_CLUSTERED,XL_CHART_TYPE.BAR_CLUSTERED,XL_CHART_TYPE.LINE,XL_CHART_TYPE.LINE_MARKERS,XL_CHART_TYPE.PIE,XL_CHART_TYPE.DOUGHNUT,XL_CHART_TYPE.AREA,XL_CHART_TYPE.RADAR]:
        deck=Presentation();s=deck.slides.add_slide(deck.slide_layouts[6]);data=CategoryChartData()
        data.categories=['A','B','第三'];data.add_series('series 1',[1,2,3]);data.add_series('series 2',[2,None,5])
        chart=s.shapes.add_chart(kind,100,100,7000000,5000000,data).chart
        chart.has_title=True;chart.chart_title.text_frame.text='native chart'
        add('resources/chart/'+kind.name,saved(deck))
    for mode,fmt in [('RGB','PNG'),('RGBA','PNG'),('L','PNG'),('LA','PNG'),('P','PNG'),('1','PNG'),('RGB','JPEG'),('RGB','GIF'),('RGB','TIFF')]:
        deck=Presentation();s=deck.slides.add_slide(deck.slide_layouts[6]);buf=BytesIO()
        color=(10,100,200,110)if mode=='RGBA'else((30,180)if mode=='LA'else((10,100,200)if mode=='RGB'else 1))
        Image.new(mode,(32,24),color).save(buf,format=fmt)
        p=s.shapes.add_picture(BytesIO(buf.getvalue()),200,200,7000000,5000000)
        p._element.find(qn('p:nvPicPr')+'/'+qn('p:cNvPr')).set('descr','  alternative <&> text  ')
        add('resources/image/'+mode+'/'+fmt,saved(deck))
    mtef=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/mtef.json.gz').read_bytes()))
    objects=[bytes.fromhex(c['hex'])for c in mtef['cases']if c['kind']=='object' and c.get('latex')][:3]
    icon=BytesIO();Image.new('RGB',(30,30),'blue').save(icon,format='PNG')
    for variant_id in ['valid-svg','bad-svg','missing-svg-rel','empty-blip','linked-blip']:
        deck=Presentation();s=deck.slides.add_slide(deck.slide_layouts[6]);pic=s.shapes.add_picture(BytesIO(icon.getvalue()),100,100,7000000,5000000)
        with ZipFile(BytesIO(saved(deck)))as z:items={n:z.read(n)for n in z.namelist()}
        root=etree.fromstring(items['ppt/slides/slide1.xml']);blip=root.find('.//'+qn('a:blip'))
        if variant_id in ['empty-blip','linked-blip']:
            blip.attrib.pop(qn('r:embed'),None)
            if variant_id=='linked-blip':blip.set(qn('r:link'),'rIdLinked')
        else:
            ns='http://schemas.microsoft.com/office/drawing/2016/SVG/main'
            ext=etree.SubElement(blip,qn('a:extLst'));ext=etree.SubElement(ext,qn('a:ext'));ext.set('uri','{SVG}')
            svg=etree.SubElement(ext,'{'+ns+'}svgBlip');svg.set(qn('r:embed'),'rIdSvg')
            if variant_id!='missing-svg-rel':
                relpath='ppt/slides/_rels/slide1.xml.rels';rels=etree.fromstring(items[relpath]);r=etree.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
                r.set('Id','rIdSvg');r.set('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/image');r.set('Target','../media/test.svg');items[relpath]=etree.tostring(rels)
                items['ppt/media/test.svg']=b'<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"><rect width="30" height="30" fill="red"/></svg>'if variant_id=='valid-svg'else b'<svg invalid'
                ct=etree.fromstring(items['[Content_Types].xml']);n=etree.SubElement(ct,'{http://schemas.openxmlformats.org/package/2006/content-types}Default');n.set('Extension','svg');n.set('ContentType','image/svg+xml');items['[Content_Types].xml']=etree.tostring(ct)
        items['ppt/slides/slide1.xml']=etree.tostring(root);out=BytesIO()
        with ZipFile(out,'w',ZIP_DEFLATED)as z:
            for n,data in items.items():z.writestr(n,data)
        add('resources/binding/'+variant_id,out.getvalue())
    for index,payload in enumerate([b'not ole',*objects]):
        for prog in ['Equation.3','MathType.Equation','Package']:
            for as_icon in [False,True]:
                deck=Presentation();s=deck.slides.add_slide(deck.slide_layouts[6])
                ole=s.shapes.add_ole_object(BytesIO(payload),prog,100,100,6000000,4000000,icon_file=BytesIO(icon.getvalue()))
                obj=ole._element.find('.//'+qn('p:oleObj'));obj.set('showAsIcon',str(int(as_icon)))
                add('resources/ole/'+str(index)+'/'+prog+'/'+str(as_icon),saved(deck))

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = 'Title inherited from layout'
    slide.placeholders[1].text = 'Subtitle inherited from layout'
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(20, 40, 60, 80).text = 'outer group'
    inner = group.shapes.add_group_shape()
    inner.shapes.add_textbox(50, 100, 70, 130).text = 'nested math/text'
    picture = BytesIO(); Image.new('RGB', (20, 20), 'red').save(picture, format='PNG')
    slide.shapes.add_picture(BytesIO(picture.getvalue()), 0, 0, prs.slide_width, prs.slide_height)
    slide.shapes.add_textbox(0, 0, prs.slide_width//2, prs.slide_height//2).text = 'covers background'
    slide.shapes.add_picture(BytesIO(picture.getvalue()), 10, 10, 10, 10)
    slide.notes_slide.notes_text_frame.text = 'speaker notes'
    blank = prs.slides.add_slide(prs.slide_layouts[6])
    blank.shapes.add_textbox(100, 300, 500, 70).text = 'left column'
    blank.shapes.add_textbox(1000, 100, 500, 70).text = 'right column'
    buf = BytesIO(); prs.save(buf)
    with ZipFile(BytesIO(buf.getvalue())) as z:
        original = {n:z.read(n) for n in z.namelist()}

    def variant(name, mutate):
        parts = dict(original)
        root = etree.fromstring(parts['ppt/slides/slide1.xml'])
        mutate(root, parts)
        parts['ppt/slides/slide1.xml'] = etree.tostring(root)
        out = BytesIO()
        with ZipFile(out, 'w', ZIP_DEFLATED) as z:
            for n,data in parts.items(): z.writestr(n,data)
        add(name, out.getvalue())

    variant('generated/base', lambda root,parts:None)
    for attr, values in [('cx', ['0','-4','200']), ('cy',['0','-5','800'])]:
        for value in values:
            def change(root, parts, attr=attr, value=value):
                root.find('.//'+qn('p:grpSp')+'/'+qn('p:grpSpPr')+'/'+qn('a:xfrm')+'/'+qn('a:chExt')).set(attr,value)
            variant('group/child-extent/'+attr+'/'+value, change)
    for field in ['off','ext','chOff','chExt']:
        def change(root,parts,field=field):
            x=root.find('.//'+qn('p:grpSp')+'/'+qn('p:grpSpPr')+'/'+qn('a:xfrm'))
            x.remove(x.find(qn('a:'+field)))
        variant('group/missing/'+field, change)
    for attr,value in [('rot','5400000'),('flipH','1'),('flipV','1')]:
        def change(root,parts,attr=attr,value=value):
            root.find('.//'+qn('p:grpSp')+'/'+qn('p:grpSpPr')+'/'+qn('a:xfrm')).set(attr,value)
        variant('group/'+attr,change)
    for idx in ['0','1','77']:
        def change(root,parts,idx=idx):
            root.find('.//'+qn('p:ph')).set('idx',idx)
        variant('placeholder/idx/'+idx,change)
    for branch in ['Choice','Fallback','EmptyChoice']:
        def change(root,parts,branch=branch):
            tree=root.find('.//'+qn('p:spTree')); shapes=list(tree.findall(qn('p:sp')))
            alt=etree.Element(qn('ve:AlternateContent'))
            target=etree.SubElement(alt,qn('ve:'+('Choice' if branch=='EmptyChoice' else branch)))
            if branch=='EmptyChoice':target=etree.SubElement(alt,qn('ve:Fallback'))
            for shape in shapes:target.append(shape)
            tree.append(alt)
        variant('alternate/'+branch,change)
    def no_size(root,parts):
        p=etree.fromstring(parts['ppt/presentation.xml']);p.remove(p.find(qn('p:sldSz')))
        parts['ppt/presentation.xml']=etree.tostring(p)
    variant('presentation/no-size',no_size)
    def reverse(root,parts):
        p=etree.fromstring(parts['ppt/presentation.xml']);ids=p.find(qn('p:sldIdLst'))
        ids[:]=list(ids)[::-1];parts['ppt/presentation.xml']=etree.tostring(p)
    variant('presentation/reverse-ids',reverse)
    for kind in ['body','sldImg','sldNum','dt','ftr','title','subTitle']:
        def change(root,parts,kind=kind):
            path='ppt/notesSlides/notesSlide1.xml';notes=etree.fromstring(parts[path])
            ph=next(n for n in notes.iter(qn('p:ph'))if n.get('type')=='body');ph.set('type',kind)
            parts[path]=etree.tostring(notes)
        variant('notes/placeholder/'+kind,change)
    for order in ['notes-only-first','blank-last','all-blank']:
        deck=Presentation();first=deck.slides.add_slide(deck.slide_layouts[6]);first.notes_slide.notes_text_frame.text='notes only'
        if order!='all-blank':
            title=deck.slides.add_slide(deck.slide_layouts[0]);title.shapes.title.text='first visible title'
        if order=='blank-last':deck.slides.add_slide(deck.slide_layouts[6])
        add('notes/'+order,saved(deck))
    for mode in ['same-link','different-link','unsafe-link','field','soft-break','whitespace','mixed-style']:
        def change(root,parts,mode=mode):
            p=root.find('.//'+qn('p:sp')+'/'+qn('p:txBody')+'/'+qn('a:p'))
            for n in list(p):p.remove(n)
            for i,text in enumerate(['  first',' ', '第二段','  ']):
                tag='a:fld' if mode=='field' and i==2 else 'a:r'
                run=etree.SubElement(p,qn(tag));rpr=etree.SubElement(run,qn('a:rPr'))
                if mode=='mixed-style':rpr.set('b',str(i%2));rpr.set('u','sng'if i==2 else'none')
                if mode.endswith('link'):
                    h=etree.SubElement(rpr,qn('a:hlinkClick'));h.set(qn('r:id'),'rIdLink'+str(i%2 if mode=='different-link' else 0))
                etree.SubElement(run,qn('a:t')).text=text
                if mode=='soft-break':etree.SubElement(p,qn('a:br'))
            if mode.endswith('link'):
                path='ppt/slides/_rels/slide1.xml.rels';rels=etree.fromstring(parts[path])
                for i in range(2):
                    r=etree.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')
                    r.set('Id','rIdLink'+str(i));r.set('Type','http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink')
                    r.set('Target','javascript:alert(1)'if mode=='unsafe-link'else'https://example.com/'+str(i));r.set('TargetMode','External')
                parts[path]=etree.tostring(rels)
        variant('text/'+mode,change)
    for source in ['run','paragraph','end','shape','layout','master']:
        for attr,values in [('b',['1','0','true','none','bogus']),('i',['on','off']),('u',['sng','none']),('strike',['dblStrike','noStrike']),('sz',['1045','1055','2400'])]:
            for value in values:
                def change(root,parts,source=source,attr=attr,value=value):
                    shape=root.find('.//'+qn('p:sp'))
                    if source in ['layout','master']:
                        path='ppt/slideLayouts/slideLayout1.xml' if source=='layout' else 'ppt/slideMasters/slideMaster1.xml'
                        other=etree.fromstring(parts[path])
                        if source=='layout':
                            tx=other.find('.//'+qn('p:sp')+'/'+qn('p:txBody')); style=tx.find(qn('a:lstStyle'))
                        else:style=other.find(qn('p:txStyles')+'/'+qn('p:titleStyle'))
                        ppr=style.find(qn('a:lvl1pPr'))
                        if ppr is None:ppr=etree.SubElement(style,qn('a:lvl1pPr'))
                    else:
                        tx=shape.find(qn('p:txBody'));p=tx.find(qn('a:p'))
                        if source=='run':
                            run=p.find(qn('a:r'));rpr=run.find(qn('a:rPr'))
                            if rpr is None:rpr=etree.SubElement(run,qn('a:rPr'))
                        elif source=='end':
                            rpr=p.find(qn('a:endParaRPr'))
                            if rpr is None:rpr=etree.SubElement(p,qn('a:endParaRPr'))
                        else:
                            parent=p if source=='paragraph' else tx.find(qn('a:lstStyle'))
                            tag='a:pPr' if source=='paragraph' else 'a:lvl1pPr'
                            ppr=parent.find(qn(tag))
                            if ppr is None:ppr=etree.SubElement(parent,qn(tag))
                    if source not in ['run','end']:
                        rpr=ppr.find(qn('a:defRPr'))
                        if rpr is None:rpr=etree.SubElement(ppr,qn('a:defRPr'))
                    rpr.set(attr,value)
                    if source in ['layout','master']:parts[path]=etree.tostring(other)
                variant('font/'+source+'/'+attr+'/'+value,change)
        for kind in ['buNone','buChar','buAutoNum','buBlip']:
            def change(root,parts,source=source,kind=kind):
                # Text-body markers, layout/master conflicts and paragraph
                # restarts deliberately follow the original inheritance chain.
                target=root;path=None
                if source in ['layout','master']:
                    path='ppt/slideLayouts/slideLayout1.xml' if source=='layout' else 'ppt/slideMasters/slideMaster1.xml'
                    target=etree.fromstring(parts[path])
                if source=='master':parent=target.find(qn('p:txStyles')+'/'+qn('p:titleStyle'));tag='a:lvl1pPr'
                else:
                    tx=target.find('.//'+qn('p:sp')+'/'+qn('p:txBody'))
                    parent=tx.find(qn('a:lstStyle')) if source in ['shape','layout'] else tx.find(qn('a:p'))
                    tag='a:lvl1pPr' if source in ['shape','layout'] else 'a:pPr'
                ppr=parent.find(qn(tag))
                if ppr is None:ppr=etree.SubElement(parent,qn(tag))
                for child in list(ppr):
                    if child.tag in [qn('a:'+k)for k in ['buNone','buChar','buAutoNum','buBlip']]:ppr.remove(child)
                bullet=etree.SubElement(ppr,qn('a:'+kind))
                if kind=='buChar':bullet.set('char','•')
                if kind=='buAutoNum':bullet.set('type','arabicPeriod');bullet.set('startAt','123456789012345678901234567890')
                if path:parts[path]=etree.tostring(target)
            variant('marker/'+source+'/'+kind,change)
    rng=random.Random(734022)
    for i in range(60):
        sequence=[(rng.randrange(0,6),rng.choice(['buNone','buChar','buAutoNum','buBlip']),rng.choice([None,1,2,5,12345678901234567890]),rng.choice(['item','  item  ','']))for j in range(rng.randrange(3,20))]
        def change(root,parts,sequence=sequence):
            body=root.find('.//'+qn('p:sp')+'/'+qn('p:txBody'))
            for p in list(body.findall(qn('a:p'))):body.remove(p)
            for level,kind,start,text in sequence:
                p=etree.SubElement(body,qn('a:p'));ppr=etree.SubElement(p,qn('a:pPr'));ppr.set('lvl',str(level))
                marker=etree.SubElement(ppr,qn('a:'+kind))
                if kind=='buChar':marker.set('char','•')
                if kind=='buAutoNum':
                    marker.set('type','arabicPeriod')
                    if start is not None:marker.set('startAt',str(start))
                run=etree.SubElement(p,qn('a:r'));etree.SubElement(run,qn('a:t')).text=text
        variant('list-sequence/'+str(i),change)
    engine=PptxConverter()
    queries=[]
    for i in range(240):
        boxes=[]
        for j in range(rng.randrange(0,55)):
            x=rng.choice([0.,100.,500.,rng.uniform(-40,1000)])
            y=rng.uniform(-40,1200)
            boxes.append([x,y,x+rng.choice([1.,4.,40.,120.,450.,1100.]),y+rng.choice([1.,30.,100.,700.])])
        if i%5==0:boxes.extend([None,[1.,1.,1.,3.],[2.,3.,1.,5.]])
        entries=[dict(bbox=b,index=j)for j,b in enumerate(boxes)]
        picture_box=boxes[0] if boxes and boxes[0] else None
        later=[(b,rng.choice([True,False]))for b in boxes[1:] if b and b[2]>b[0] and b[3]>b[1]]
        from types import SimpleNamespace
        text_shapes=[SimpleNamespace(bbox=b,shape=SimpleNamespace(has_text_frame=text,_element=etree.fromstring('<p:sp xmlns:p="'+engine.namespaces['p']+'" xmlns:a="'+engine.namespaces['a']+'"><a:t>text</a:t></p:sp>'))) for b,text in later]
        queries.append(dict(boxes=boxes,order=[e['index']for e in sort_entries(entries,beta=2.,density_threshold=.9)],
            union=engine._rectangles_union_area([b for b,text in later]),picture=picture_box,later=later,
            small=engine._is_small_picture(picture_box,1200,900),
            background=engine._is_background_picture(SimpleNamespace(bbox=picture_box),text_shapes)))
    target=ROOT/'tests/fixtures/office_rust_reference_068/pptx_context.json.gz'
    fonts=[]
    from types import SimpleNamespace
    for size in list(range(-100,10001,5))+[1065,1075,100000,99999999]:
        run=etree.fromstring(('<a:r xmlns:a="'+engine.namespaces['a']+'"><a:rPr sz="'+str(size)+'"/><a:t>x</a:t></a:r>').encode())
        fake=SimpleNamespace(_r=run)
        fonts.append(dict(xml=etree.tostring(run).decode(),styles=[],font_size_pt=engine._resolve_effective_run_font_size_pt(fake,[])))
    for attr in ['b','i','u','strike']:
        for value in ['1','0','true','false','t','f','on','off','none','noStrike','sng','bogus',' TRUE ']:
            run=etree.fromstring(('<a:r xmlns:a="'+engine.namespaces['a']+'"><a:rPr '+attr+'="'+value+'"/><a:t>x</a:t></a:r>').encode())
            styles=engine._get_style_str_from_run(SimpleNamespace(_r=run),[])
            fonts.append(dict(xml=etree.tostring(run).decode(),styles=styles.split(',')if styles else[],font_size_pt=None))
    import copy
    titles=[]
    for i in range(400):
        blocks=[]
        for j in range(rng.randrange(0,25)):
            role=rng.choice([None,None,None,'center_title','title','subtitle'])
            b=dict(type='text',content='block '+str(j),_effective_font_size_pt=rng.choice([None,10.,12.,14.,16.,18.,20.,24.,32.]),
                _effective_all_bold=rng.choice([True,False]))
            if role:b.update(_pptx_title_candidate=True,_pptx_title_role=role,level=2)
            blocks.append(b)
        for first in [True,False]:
            expected=copy.deepcopy(blocks);engine._promote_slide_text_blocks_to_titles(expected)
            engine._finalize_slide_title_types(expected,is_first_visible_slide=first);engine._cleanup_slide_text_block_metadata(expected)
            titles.append(dict(blocks=blocks,first=first,expected=expected))
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,queries=queries,fonts=fonts,titles=titles),ensure_ascii=True).encode(),mtime=0))
    print(len(cases),'PPTX packages;',sum('error'in c for c in cases),'original errors;',len(queries),'geometry/order;',len(fonts),'font;',len(titles),'title queries')


if __name__ == '__main__': main()
