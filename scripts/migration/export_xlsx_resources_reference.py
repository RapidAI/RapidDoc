"""Freeze original XLSX drawing/chart/equation arbitration, including real files.

Encoded PNG bytes may differ; only replace data URIs after checking decoded
RGBA dimensions and SHA256. All other block content/order remains exact.
"""
import base64
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/native_office')]


def normalize(value):
    from PIL import Image
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    def replace(match):
        uri = match.group()
        header, encoded = uri.split(',', 1)
        data = base64.b64decode(encoded)
        if header == 'data:image/svg+xml;base64':
            return 'verified-svg:' + hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as image:
            rgba = image.convert('RGBA')
            return f'verified-image:{header}:{image.width}x{image.height}:' + hashlib.sha256(rgba.tobytes()).hexdigest()

    return re.sub(r'data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+', replace, value)


def main():
    from loguru import logger
    from openpyxl import Workbook
    from openpyxl.chart import (BarChart, PieChart, ScatterChart, BubbleChart, LineChart, Reference, Series,
                               AreaChart, AreaChart3D, LineChart3D, StockChart, RadarChart,
                               PieChart3D, DoughnutChart, BarChart3D, ProjectedPieChart,
                               SurfaceChart, SurfaceChart3D)
    from openpyxl.drawing.image import Image as SheetImage
    from openpyxl.drawing.spreadsheet_drawing import AbsoluteAnchor, OneCellAnchor, TwoCellAnchor, AnchorMarker
    from PIL import Image
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xlsx.xlsx_converter import XlsxConverter
    import _image_mtef_test_utils as image_math
    import _mtef_test_utils as v3
    logger.remove()
    cases = []

    def capture(name, data):
        case = dict(name=name, package_hex=data.hex(), queries=[])
        for hidden, gap, singleton in [(False, None, True), (True, 0, False)]:
            engine = XlsxConverter(include_hidden_sheets=hidden, gap_tolerance=gap, treat_singleton_as_text=singleton)
            q = dict(include_hidden=hidden, gap=gap, singleton=singleton)
            try:
                engine.convert(BytesIO(data))
                q['pages'] = normalize(engine.pages)
            except Exception as error:
                q['error'] = type(error).__name__
            case['queries'].append(q)
        cases.append(case)

    def save(book):
        out = BytesIO()
        book.save(out)
        return out.getvalue()

    def rewrite(data, updates):
        out = BytesIO()
        with ZipFile(BytesIO(data)) as source, ZipFile(out, 'w', ZIP_DEFLATED) as target:
            for name in source.namelist():
                content = source.read(name)
                change = updates.get(name)
                if change is None:
                    target.writestr(name, content)
                elif callable(change):
                    target.writestr(name, change(content.decode()).encode())
                else:
                    target.writestr(name, change)
            for name, content in updates.items():
                if name not in source.namelist():
                    target.writestr(name, content)
        return out.getvalue()

    def bitmap(fmt='PNG', color='red'):
        out = BytesIO()
        Image.new('RGB', (13, 9), color).save(out, fmt)
        return out.getvalue()

    def book():
        wb = Workbook()
        ws = wb.active
        ws.title = 'Data'
        ws.append(['key', 'one', 'two'])
        ws.append(['A', 1, 3])
        ws.append(['B', 2, 4])
        return wb

    for fmt in ['PNG', 'BMP', 'TIFF', 'JPEG', 'GIF', 'WEBP']:
        for placement in ['table', 'separate', 'same', 'absolute', 'merged']:
            wb = book()
            if placement == 'merged':
                wb.active.merge_cells('A1:C2')
            for i in range(2):
                image = SheetImage(BytesIO(bitmap('PNG', ['red', 'blue'][i])))
                image.anchor = AbsoluteAnchor() if placement == 'absolute' else ('B2' if placement in ['table', 'same', 'merged'] else ['E8', 'G11'][i])
                wb.active.add_image(image)
            data = save(wb)
            data = rewrite(data, {f'xl/media/image{i+1}.png': bitmap(fmt, color) for i, color in enumerate(['red', 'blue'])})
            capture(f'images {fmt} {placement}', data)

    wb = book()
    for i, anchor in enumerate([TwoCellAnchor(_from=AnchorMarker(row=1, col=1)), AbsoluteAnchor(), OneCellAnchor(_from=AnchorMarker(row=8, col=4))]):
        im = SheetImage(BytesIO(bitmap()))
        im.anchor = anchor
        wb.active.add_image(im)
    capture('same pixels distinct anchors', save(wb))

    for chart_type in [BarChart, PieChart, LineChart, ScatterChart, BubbleChart, AreaChart, AreaChart3D,
                       LineChart3D, StockChart, RadarChart, PieChart3D, DoughnutChart, BarChart3D,
                       ProjectedPieChart, SurfaceChart, SurfaceChart3D]:
        wb = book()
        chart = chart_type()
        if chart_type in [ScatterChart, BubbleChart]:
            chart.series.append(Series(Reference(wb.active, min_col=2, min_row=2, max_row=3), Reference(wb.active, min_col=3, min_row=2, max_row=3)))
        else:
            chart.add_data(Reference(wb.active, min_col=2, max_col=3, min_row=1, max_row=3), titles_from_data=True)
            chart.set_categories(Reference(wb.active, min_col=1, min_row=2, max_row=3))
        wb.active.add_chart(chart, 'E6')
        data = save(wb)
        capture('chart ' + chart_type.__name__, data)
        capture('unreferenced chart ' + chart_type.__name__, rewrite(data, {'xl/worksheets/sheet1.xml': lambda s: re.sub(r'<drawing\b[^>]*/>', '', s)}))
        if chart_type == BarChart:
            for name, transform in [
                ('cross-sheet', lambda s: s.replace("'Data'!", "'Other'!")),
                ('prefix equals', lambda s: s.replace("<f>'Data'!", "<f>='Data'!")),
                ('trailing expression', lambda s: s.replace('</f>', '+0</f>')),
                ('whole column', lambda s: s.replace('$B$2:$B$3', '$B:$B')),
                ('discontinuous', lambda s: s.replace('$B$1', '$E$1').replace('$B$2:$B$3', '$E$7:$E$8')),
            ]:
                capture('chart ' + name, rewrite(data, {'xl/charts/chart1.xml': transform}))
            other = LineChart()
            other.add_data(Reference(wb.active, min_col=3, min_row=1, max_row=3), titles_from_data=True)
            chart += other
            capture('chart combo first plot only', save(wb))

    # Raw drawing, shared by OMML, image-comment MTEF, OLE previews and DISPIMG.
    wb = book()
    im = SheetImage(BytesIO(bitmap()))
    im.anchor = 'B2'
    wb.active.add_image(im)
    base = save(wb)
    mtef = image_math.build_gif_with_mtef(v3.mtef_equation(v3.mtef_text('x+y')))
    for placement in ['B2', 'E8']:
        wb = book()
        im = SheetImage(BytesIO(mtef))
        wb.active.add_image(im, placement)
        capture('image MTEF ' + placement, save(wb))
    math = '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:r><m:t>x+z</m:t></m:r></m:oMath>'
    for pos in ['table', 'standalone']:
        capture('OMML ' + pos, rewrite(base, {'xl/drawings/drawing1.xml': lambda s, pos=pos: s.replace('</pic>', math + '</pic>').replace('<row>1</row>', '<row>8</row>' if pos == 'standalone' else '<row>1</row>')}))

    native = v3.equation_native(v3.mtef_equation(v3.mtef_text('a+b')))
    ole = v3._build_nested_cfb([dict(name='Root Entry', kind=5, child=1), dict(name='Equation Native', kind=2, data=native)])
    rns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    for mode in ['native', 'bad', 'icon', 'linked', 'image-mtef', 'omml', 'standalone', 'no-anchor', 'vml']:
        anchor = '<anchor><from><row>1</row><col>1</col></from></anchor>'
        if mode == 'standalone':
            anchor = anchor.replace('<row>1', '<row>8')
        if mode in ['no-anchor', 'vml']:
            anchor = ''
        obj = f'<oleObjects><oleObject progId="Equation.3" shapeId="1" xmlns:r="{rns}" r:id="ole"' + (' dvAspect="DVASPECT_ICON"' if mode == 'icon' else ' link="external"' if mode == 'linked' else '') + f'><objectPr r:id="preview">{anchor}</objectPr></oleObject></oleObjects>'
        rels = f'<Relationship Type="{rns}/oleObject" Target="../embeddings/formula.bin" Id="ole"/><Relationship Type="{rns}/image" Target="../media/image1.png" Id="preview"/>'
        updates = {
            'xl/worksheets/sheet1.xml': lambda s, obj=obj: s.replace('</worksheet>', obj + '</worksheet>'),
            'xl/worksheets/_rels/sheet1.xml.rels': lambda s, rels=rels: s.replace('</Relationships>', rels + '</Relationships>'),
            'xl/embeddings/formula.bin': ole if mode not in ['bad', 'image-mtef', 'vml'] else b'invalid OLE',
        }
        if mode == 'image-mtef':
            updates['xl/media/image1.png'] = mtef
        if mode == 'omml':
            updates['xl/drawings/drawing1.xml'] = lambda s: s.replace('</pic>', math + '</pic>')
        if mode == 'no-anchor':
            updates['xl/drawings/drawing1.xml'] = lambda s: s.replace('id="1"', 'id="7"')
        if mode == 'vml':
            updates['xl/drawings/drawing1.xml'] = lambda s: s.replace('id="1"', 'id="7"')
            updates['xl/worksheets/_rels/sheet1.xml.rels'] = lambda s, rels=rels: s.replace('</Relationships>', rels + f'<Relationship Type="{rns}/vmlDrawing" Target="../drawings/vml1.vml" Id="vml"/></Relationships>')
            updates['xl/drawings/vml1.vml'] = '<xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:x="urn:schemas-microsoft-com:office:excel"><v:shape id="_x0000_s1"><x:ClientData><x:Anchor>4, 0, 8, 0, 6, 0, 10, 0</x:Anchor></x:ClientData></v:shape></xml>'
        capture('OLE ' + mode, rewrite(base, updates))

    for mode in ['png', 'gif-mtef', 'missing', 'absolute', 'repeated']:
        wb = book()
        wb.active['B2'] = 'DISPIMG("image-key",1)'
        if mode == 'repeated':
            wb.active['B3'] = 'DISPIMG("image-key",1)'
        payload = save(wb)
        capture('DISPIMG ' + mode, rewrite(payload, {
            'xl/cellimages.xml': f'<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData" xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="{rns}"><etc:cellImage><xdr:cNvPr name="image-key"/><a:blip r:embed="im"/></etc:cellImage></etc:cellImages>',
            'xl/_rels/cellimages.xml.rels': f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="im" Type="{rns}/image" Target="'+('/xl/media/cell.png' if mode == 'absolute' else 'media/missing.png' if mode == 'missing' else 'media/cell.png')+'"/></Relationships>',
            'xl/media/cell.png': mtef if mode == 'gif-mtef' else bitmap(),
        }))
    for path in sorted((ROOT / 'demo/xlsx').glob('*.xlsx')) + [ROOT / 'demo/ms_office_docs/xlsx_01.xlsx']:
        capture('real ' + path.name, path.read_bytes())
    dest = ROOT / 'tests/fixtures/office_rust_reference_068/xlsx_resources.json.gz'
    dest.write_bytes(gzip.compress(json.dumps(dict(cases=cases), ensure_ascii=True).encode(), mtime=0))
    print('XLSX resources:', len(cases), 'workbooks;', sum(len(c['queries']) for c in cases), 'queries;', sum('error' in q for c in cases for q in c['queries']), 'errors')


if __name__ == '__main__':
    main()
