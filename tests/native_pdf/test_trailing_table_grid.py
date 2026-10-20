from io import BytesIO
from types import SimpleNamespace

import pytest
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableRule


def physical_source(*, end=170, extra=None, footer=False):
    rules=[NativeTableRule((20,y,280,y),.7,'horizontal') for y in (30,100,135,170)]
    rules.extend(NativeTableRule((x,30,x,170),.7,'vertical') for x in (20,80,280))
    # This separator is absent from the merged header but present in the body.
    rules.append(NativeTableRule((150,100,150,end),.7,'vertical'))
    if extra:
        rules.append(extra)
    source=SimpleNamespace(page_index=0,page_size=(300,240),drawing_lines=rules,
                           lines=[SimpleNamespace(effective_height=10)])
    block={'type':'table','bbox':(20,30,280,170)}
    prepared=SimpleNamespace(fixed_blocks=[block],remaining_lines=[])
    if footer:
        prepared.remaining_lines=[_LineItem('Separate paragraph',(20,180,210,192),0,0,effective_height=10)]
    return source,prepared


def test_trailing_tracks_include_separator_missing_from_merged_header():
    source,page=physical_source()
    grid=_trailing_table_continuation_grid(source,page)
    assert grid and grid.positions == pytest.approx(tuple(x/300 for x in (20,80,150,280)))


@pytest.mark.parametrize('end',[140,150,164,167])
def test_separator_not_reaching_page_end_is_not_exported(end):
    source,page=physical_source(end=end)
    grid=_trailing_table_continuation_grid(source,page)
    assert grid and len(grid.positions)==3


def test_short_terminal_decoration_does_not_create_column():
    source,page=physical_source(extra=NativeTableRule((210,167,210,170),.7,'vertical'))
    grid=_trailing_table_continuation_grid(source,page)
    assert grid and len(grid.positions)==4


def test_fragmented_terminal_separator_uses_union_coverage():
    source,page=physical_source(end=150)
    source.drawing_lines.append(NativeTableRule((150,151,150,170),.7,'vertical'))
    grid=_trailing_table_continuation_grid(source,page)
    assert grid and len(grid.positions)==4


def test_text_after_table_cancels_continuation_hint():
    source,page=physical_source(footer=True)
    assert _trailing_table_continuation_grid(source,page) is None


def continued_pdf(scale=1, next_separator=150, heading=False, note=False, header_end=100):
    stream=BytesIO()
    canvas=Canvas(stream,pagesize=(300*scale,240*scale))
    def begin():
        canvas.scale(scale,scale)
        canvas.setFont('Helvetica',9)
    def line(x0,y0,x1,y1):
        canvas.line(x0,240-y0,x1,240-y1)
    def text(x,y,t):
        canvas.drawString(x,240-y,t)
    begin()
    for y in (30,header_end,135,170):
        line(20,y,280,y)
    for x in (20,80,280):
        line(x,30,x,170)
    line(150,header_end,150,170)
    text(25,50,'Key')
    text(90,50,'Merged upper cells')
    for y in (header_end+15,155):
        for x,t in [(25,'Entry'),(85,'Detail'),(160,'Process value')]:
            text(x,y,t)
    if note:
        text(20,193,'Independent paragraph after table.')
    canvas.showPage()
    begin()
    if heading:
        text(20,17,'Independent section starts here')
    for y in (30,170):
        line(20,y,280,y)
    for x in (20,80,next_separator,280):
        line(x,30,x,170)
    for i in range(8):
        text(next_separator+5,48+i*14,f'Value {i} remains in this cell.')
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize('scale',[.7,1,1.8])
@pytest.mark.parametrize('header_end',[75,100])
def test_native_layout_retains_single_populated_column_continuation(scale,header_end):
    pages=analyze_docvortex_layout(continued_pdf(scale=scale,header_end=header_end))
    tables=[b for b in pages[1] if b['type']=='table']
    assert len(tables)==1
    assert 'Value 0' in str(tables[0]['content'])
    assert 'Value 7' in str(tables[0]['content'])


@pytest.mark.parametrize('options',[{'next_separator':165},{'heading':True},{'note':True}])
def test_unrelated_or_misaligned_single_column_frame_is_not_inherited(options):
    pages=analyze_docvortex_layout(continued_pdf(**options))
    assert not any(b['type']=='table' for b in pages[1])
    assert 'Value 0' in str(pages[1])
