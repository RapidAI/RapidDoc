import pypdfium2 as pdfium
from reportlab.pdfgen.canvas import Canvas
from io import BytesIO

from rapid_doc.model.table.rule_table import build_rule_table
from rapid_doc.utils.pdf_text_tool import get_page, get_page_vector_lines


def test_reportlab_table_round_trips_through_pdfium():
    stream=BytesIO()
    canvas=Canvas(stream, pagesize=(200,160))
    # ReportLab/PDF use bottom-left coordinates. The extracted bbox is
    # [20,20,170,110] after conversion to RapidDoc's top-left coordinates.
    for x in (20,70,120,170): canvas.line(x,50,x,140)
    for y in (50,80,110,140): canvas.line(20,y,170,y)
    for r in range(3):
        for c in range(3): canvas.drawString(25+c*50,125-r*30,f'{r}{c}')
    canvas.save()

    doc=pdfium.PdfDocument(stream.getvalue()); page=doc[0]
    page_dict=get_page(page)
    page_dict['vector_lines']=get_page_vector_lines(page)
    result=build_rule_table(page_dict, [20,20,170,110])
    page.close(); doc.close()
    assert result is not None
    assert result.rows == 3 and result.columns == 3
    assert '<td>22</td>' in result.html


def test_pdftext_chars_are_materialized_for_rule_table_consumers():
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(100, 100))
    canvas.drawString(10, 80, "AB")
    canvas.save()

    doc = pdfium.PdfDocument(stream.getvalue())
    page = doc[0]
    page_dict = get_page(page)
    page.close()
    doc.close()

    assert isinstance(page_dict["chars"], list)
    assert "".join(char["char"] for char in page_dict["chars"] if char["char"].strip()) == "AB"
    assert all("bbox" in char and "rotation" in char for char in page_dict["chars"])


def test_filled_thin_rectangles_are_treated_as_rules():
    stream=BytesIO(); canvas=Canvas(stream, pagesize=(200,160))
    for x in (20,70,120,170): canvas.rect(x,50,.48,90,stroke=0,fill=1)
    for y in (50,80,110,140): canvas.rect(20,y,150,.48,stroke=0,fill=1)
    for r in range(3):
        for c in range(3): canvas.drawString(25+c*50,125-r*30,f'{r}{c}')
    canvas.save()
    doc=pdfium.PdfDocument(stream.getvalue()); page=doc[0]
    page_dict=get_page(page); page_dict['vector_lines']=get_page_vector_lines(page)
    result=build_rule_table(page_dict,[20,19.5,170.5,110.5])
    page.close(); doc.close()
    assert result is not None
    assert '<td>22</td>' in result.html
