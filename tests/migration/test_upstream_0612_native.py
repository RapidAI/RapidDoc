"""Installed native paths versus independently updated Python rules."""
import json
from dataclasses import asdict
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"native_pdf"))
import test_upstream_0612_unicode as unicode_cases
from test_upstream_069_banded_rowspan import _paired_input
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import recover_native_pdf_table
from rapid_doc_rust import legacy_pipeline_json, read_pdf_character_primitives

@pytest.mark.parametrize("angle",[0,90,180,270])
@pytest.mark.parametrize("scale",[.5,1,2])
def test_banded_rowspan_native_matches_python(angle,scale):
    table=_paired_input(angle=angle,scale=scale)
    expected=recover_native_pdf_table(table)
    actual=json.loads(legacy_pipeline_json("recover_native_table_primitives",json.dumps({"table_input":asdict(table)})))
    assert (actual["rows"],actual["cols"])==(expected.rows,expected.cols)
    assert actual["html"]==expected.html
    assert [(c["row"],c["col"],c["rowspan"],c["colspan"],c["content"]) for c in actual["cells"]]==[(c.row,c.col,c.rowspan,c.colspan,c.content) for c in expected.cells]

@pytest.mark.parametrize("rotation",[0,90,180,270])
@pytest.mark.parametrize("names",[("beta","similar","greaterequal"),("f_f_i","similar","NoSuchGlyph"),("uni03B2","similar","u2265"),("uni03b2","similar","greaterequal"),("unknown_beta","similar","greaterequal"),("beta_suffix","similar","greaterequal"),("uni03B2.alt","similar","greaterequal")])
def test_null_unicode_native_matches_python(rotation,names):
    data=unicode_cases._pdf(rotation=rotation,names=names)
    with PDFDocument(data) as document:
        expected=document[0].get_chars_with_geometry().chars
    actual=read_pdf_character_primitives(data)["restored_chars"]
    assert [(c["char_idx"],c["char"]) for c in actual]==[(c["char_idx"],c["char"]) for c in expected]
    assert [c["raw_code"] for c in actual]==[c["raw_code"] for c in expected]

@pytest.mark.parametrize("form",[False,True])
def test_shared_font_resources_native_match_python(form,monkeypatch):
    captured=[]
    class Capture(PDFDocument):
        def __init__(self,data,*args,**kwargs):
            captured.append(data)
            super().__init__(data,*args,**kwargs)
    monkeypatch.setattr(unicode_cases,"PDFDocument",Capture)
    unicode_cases.test_same_name_and_shared_cmap_keep_per_font_differences(form)
    data=captured[0]
    with PDFDocument(data) as document:expected=document[0].get_chars()
    actual=read_pdf_character_primitives(data)["restored_chars"]
    assert ''.join(c['char'] for c in actual)==''.join(c['char'] for c in expected)

@pytest.mark.parametrize("index",[0,1])
def test_real_embedded_font_fixture(index):
    root=Path(__file__).parents[2]
    provenance=json.loads((root/'tests/fixtures/pdf_unicode_fallback_source.json').read_text('utf8'))
    data=(root/provenance['fixture']).read_bytes()
    actual=read_pdf_character_primitives(data,page_index=index)['restored_chars']
    assert ''.join(c['char'] for c in actual)==provenance['pages'][index]['expected']

def test_real_banded_rowspan_fixture():
    import gzip
    fixture=json.loads(gzip.decompress((Path(__file__).parents[1]/'fixtures/docvortex_069_banded_rowspan.json.gz').read_bytes()))
    actual=json.loads(legacy_pipeline_json('recover_native_table_primitives',json.dumps({'table_input':fixture['table_input']})))
    assert (actual['rows'],actual['cols'])==(11,5)
    assert actual['html']==fixture['expected_html']
    assert sum(cell['rowspan']==2 for cell in actual['cells'])==5
