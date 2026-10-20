from collections import Counter
from io import BytesIO
from pathlib import Path
import subprocess
import asyncio
import xml.etree.ElementTree as ET

from PIL import Image

import pytest

from rapid_doc import RapidDoc
from rapid_doc.main import MemoryDataWriter
from rapid_doc.backend.office.office_analyze import office_analyze
from rapid_doc.backend.office.office_middle_json_mkcontent import union_make
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes, guess_suffix_by_path

FIXTURES = Path(__file__).parent / "fixtures/docvortex_migration"
CASES = [("docx_01.docx", 3), ("docx_01.doc", 3), ("xlsx_01.xlsx", 3),
         ("xlsx_01.xls", 3), ("pptx_01.pptx", 6), ("pptx_01.ppt", 6)]


def _leaves(value, structured):
    found = Counter()
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in {"text", "equation_inline", "inline_equation", "code_inline"} and isinstance(value.get("content"), str):
            key = "inline_equation" if kind == "equation_inline" else kind
            found[(key, value["content"], tuple(value.get("styles" if structured else "style", [])))] += 1
        for child in value.values():
            found.update(_leaves(child, structured))
    elif isinstance(value, list):
        for child in value:
            found.update(_leaves(child, structured))
    return found


@pytest.mark.parametrize(("filename", "pages"), CASES)
def test_six_formats_preserve_raw_text_and_assets(filename, pages, monkeypatch):
    def no_soffice(*args, **kwargs):
        raise AssertionError("Office parsing must not start a conversion process")
    monkeypatch.setattr(subprocess, "run", no_soffice)
    path = FIXTURES / filename
    assert guess_suffix_by_bytes(path.read_bytes()) == path.suffix[1:]
    assert guess_suffix_by_path(path) == path.suffix[1:]
    writer = MemoryDataWriter()
    middle, raw = office_analyze(path.read_bytes(), image_writer=writer)
    assert len(raw) == len(middle["pdf_info"]) == pages
    missing = _leaves(raw, True) - _leaves(middle["pdf_info"], False)
    assert not missing, f"Lost native inline content: {missing}"
    markdown = union_make(middle["pdf_info"], MakeMode.MM_MD, "images")
    assert markdown.strip() and writer.data
    for name, data in writer.data.items():
        assert data
        if name.endswith(".svg"):
            assert ET.fromstring(data).tag == "{http://www.w3.org/2000/svg}svg"
        else:
            with Image.open(BytesIO(data)) as image:
                image.verify()


@pytest.mark.parametrize("filename", ["docx_01.doc", "xlsx_01.xls", "pptx_01.ppt"])
@pytest.mark.parametrize("input_kind", ["path", "bytes", "stream"])
def test_legacy_routes_use_native_parsing(filename, input_kind, monkeypatch):
    engine = RapidDoc()
    monkeypatch.setattr(engine, "_parse_pipeline_batch", lambda **kw: pytest.fail("Office entered the PDF model pipeline"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected conversion"))
    path = FIXTURES / filename
    value = path if input_kind == "path" else path.read_bytes()
    if input_kind == "stream":
        value = BytesIO(value)
    output = engine(value)
    assert output.markdown and output.images
    assert output.middle_json["_backend"] == "office"
    if input_kind == "stream":
        assert not value.closed


def test_legacy_batch_and_url_routes(monkeypatch):
    import rapid_doc.main as main
    path = FIXTURES / "pptx_01.ppt"
    class Response:
        content = path.read_bytes()
        def raise_for_status(self):
            pass
    monkeypatch.setattr(main.requests, "get", lambda *a, **kw: Response())
    engine = RapidDoc()
    monkeypatch.setattr(engine, "_parse_pipeline_batch", lambda **kw: pytest.fail("Office entered PDF pipeline"))
    outputs = engine([path, "https://example.org/deck.ppt"])
    assert len(outputs) == 2 and outputs[0].markdown == outputs[1].markdown


@pytest.mark.parametrize("suffix", ["docx", "xlsx", "pptx", "xlsm"])
def test_empty_modern_office_input_retains_supported_behavior(tmp_path, suffix):
    path = tmp_path / ("empty." + suffix)
    path.write_bytes(b"")
    assert RapidDoc()(path).middle_json["_backend"] == "office"


def test_svg_data_uri_uses_correct_mime_type():
    assert RapidDoc()._to_data_uri("formula.svg", b"<svg/>").startswith("data:image/svg+xml;base64,")


@pytest.mark.parametrize("filename", ["docx_01.doc", "xlsx_01.xls", "pptx_01.ppt"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cli_and_async_api_route_legacy_office_directly(tmp_path, filename, asynchronous, monkeypatch):
    from rapid_doc.cli import common
    monkeypatch.setattr(common, "_process_pipeline", lambda *a, **kw: pytest.fail("Office entered PDF pipeline"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected conversion"))
    kwargs = dict(output_dir=str(tmp_path), pdf_file_names=["native"],
                  pdf_bytes_list=[(FIXTURES / filename).read_bytes()], p_lang_list=["ch"],
                  f_dump_orig_pdf=False, f_draw_layout_bbox=False, f_draw_span_bbox=False)
    if asynchronous:
        asyncio.run(common.aio_do_parse(**kwargs))
    else:
        common.do_parse(**kwargs)
    markdown = list(tmp_path.rglob("*.md"))
    assert len(markdown) == 1 and markdown[0].read_text(encoding="utf-8").strip()
    assert kwargs["pdf_bytes_list"] == []
