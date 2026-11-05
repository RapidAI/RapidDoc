"""A real DOCX with a limited image must retain text, image position, and later tables."""
from io import BytesIO
import struct, zipfile, base64
from pathlib import Path
import pytest
from docx import Document
from PIL import Image
from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
from rapid_doc._vendor.docvortex_core.analyzers.native.office.image import get_image_limit_placeholder_data_uri

def package(table):
    count=100000
    records=struct.pack("<IHH",4,0x102,1)*count+struct.pack("<IH",3,0)
    wmf=struct.pack("<HHHIHIH",1,9,0x300,(18+len(records))//2,0,4,0)+records
    doc=Document();doc.add_paragraph("SAFE BEFORE")
    png=BytesIO();Image.new("RGB",(2,2)).save(png,format="PNG");png.seek(0)
    if table:
        row=doc.add_table(rows=1,cols=2).rows[0];row.cells[0].paragraphs[0].add_run().add_picture(png);row.cells[1].text="SAFE CELL"
    else:doc.add_picture(png)
    doc.add_paragraph("SAFE AFTER");doc.add_table(rows=1,cols=1).cell(0,0).text="LATER TABLE"
    original=BytesIO();doc.save(original);out=BytesIO()
    with zipfile.ZipFile(original) as source,zipfile.ZipFile(out,'w') as target:
        for item in source.infolist():
            data=source.read(item)
            if item.filename.startswith('word/media/'):data=wmf
            elif item.filename=='[Content_Types].xml':data=data.replace(b'ContentType="image/png"',b'ContentType="image/x-wmf"')
            target.writestr(item,data)
    return out.getvalue()

@pytest.mark.parametrize("table",[False,True])
def test_docx_placeholder_and_remaining_content(table,tmp_path):
    data=package(table);converter=DocxConverter();converter.convert(BytesIO(data))
    text=str(converter.pages)
    assert 'SAFE BEFORE' in text and 'SAFE AFTER' in text and 'LATER TABLE' in text
    expected=base64.b64decode(get_image_limit_placeholder_data_uri().split(',')[1])
    assert get_image_limit_placeholder_data_uri() in text or base64.b64encode(expected).decode() in text
    source=tmp_path/'limited.docx';source.write_bytes(data)
    from rapid_doc_rust import RapidDoc
    parsed=RapidDoc()(source)
    assert 'SAFE BEFORE' in parsed.markdown and 'SAFE AFTER' in parsed.markdown and 'LATER TABLE' in parsed.markdown
    assert expected in parsed.images.values()
    if table:assert 'SAFE CELL' in parsed.markdown
