"""Missing font maps require observed decoding defects before selecting OCR."""

from io import BytesIO

import pypdfium2 as pdfium
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.utils import pdf_classify as classifier


def sample(fonts=None, *, errors=None, abnormal=None, page_index=0):
    fonts = fonts or {"ReadableCID": 500}
    return {
        "page_index": page_index,
        "char_count": sum(fonts.values()),
        "font_name_counts": fonts,
        "non_generated_char_count": sum(fonts.values()),
        "font_non_generated_char_counts": fonts,
        "font_non_generated_unicode_map_error_counts": errors or {},
        "font_non_generated_abnormal_char_counts": abnormal or {},
    }


def signal(samples, names=("ReadableCID",)):
    return classifier._get_cid_font_usage_signal_from_samples(
        samples,
        {"triggered": True, "page_fonts": {s["page_index"]: set(names) for s in samples}},
    )


def text_pdf():
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 500))
    for row in range(12):
        canvas.drawString(20, 470 - row * 30, "Readable native text without decoding defects.")
    canvas.save()
    return stream.getvalue()


def cid_pdf(extra_cid=""):
    """Build Identity-H/Adobe-GB1 text without ToUnicode or a local font dependency."""
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 500))
    canvas.setFont("STSong-Light", 12)
    canvas.drawString(20, 450, "ABC")
    canvas.save()
    reader = PdfReader(stream)
    page = reader.pages[0]
    fonts = page["/Resources"]["/Font"]
    font_key = next(key for key, font in fonts.items() if font.get_object()["/Subtype"] == "/Type0")
    fonts[font_key][NameObject("/Encoding")] = NameObject("/Identity-H")
    # Adobe-GB1 CIDs 34/35/36/1 decode to A/B/C/space via the standard CMap.
    row = ("002200230024" + extra_cid + "0001").encode() * 20
    content = DecodedStreamObject()
    content.set_data(b"BT " + font_key.encode() + b" 12 Tf 20 450 Td 0 -20 TD "
                     + (b"<" + row + b"> Tj T* ") * 6 + b"ET")
    page[NameObject("/Contents")] = content
    writer = PdfWriter()
    writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_real_missing_map_can_decode_through_standard_cmap():
    data = cid_pdf()
    assert classifier._get_font_resource_signals_pypdf(data, [0])["cid_without_to_unicode"]["triggered"]
    assert classifier.classify(data) == "txt"


@pytest.mark.parametrize("cid,field", [
    ("FFFF", "font_non_generated_unicode_map_error_counts"),
    ("0000", "font_non_generated_abnormal_char_counts"),
])
def test_real_cid_decoding_failures_remain_ocr(cid, field):
    data = cid_pdf(cid)
    with pdfium.PdfDocument(data) as document:
        values = classifier._collect_pdfium_text_samples(document, [0])
    resources = classifier._get_font_resource_signals_pypdf(data, [0])["cid_without_to_unicode"]
    assert values[0][field]["STSong-Light"] >= 30
    assert classifier._get_cid_font_usage_signal_from_samples(values, resources)["triggered"]
    assert classifier.classify(data) == "ocr"


def test_missing_map_with_readable_characters_stays_text():
    assert not signal([sample()])["triggered"]


@pytest.mark.parametrize("field,count", [("errors", 4), ("abnormal", 3)])
def test_local_font_defects_are_not_diluted_by_other_fonts(field, count):
    value = sample({"ReadableCID": 100, "Other": 4900}, **{field: {"ReadableCID": count}})
    assert signal([value])["triggered"]


@pytest.mark.parametrize("field,count", [("errors", 3), ("abnormal", 2)])
def test_few_isolated_defects_do_not_force_ocr(field, count):
    value = sample({"ReadableCID": 100}, **{field: {"ReadableCID": count}})
    assert not signal([value])["triggered"]


def test_errors_in_another_font_do_not_condemn_readable_cid_font():
    value = sample({"ReadableCID": 100, "Other": 100}, errors={"Other": 100}, abnormal={"Other": 100})
    assert not signal([value])["triggered"]


def test_good_cid_fonts_do_not_dilute_a_bad_cid_font():
    value = sample({"ReadableCID": 4900, "BadCID": 100}, errors={"BadCID": 10})
    assert signal([value], names=("ReadableCID", "BadCID"))["triggered"]


def test_readable_first_page_does_not_hide_bad_later_page():
    values = [sample(), sample(errors={"ReadableCID": 50}, page_index=1)]
    result = signal(values)
    assert result["triggered"]
    assert result["page_index"] == 1


@pytest.mark.parametrize("fonts", [{"ReadableCID": 29}, {"ReadableCID": 30, "Other": 5000}])
def test_tiny_font_fragments_remain_below_usage_threshold(fonts):
    assert not signal([sample(fonts, errors={"ReadableCID": fonts["ReadableCID"]})])["triggered"]


def test_unused_cid_resource_does_not_force_ocr():
    assert not signal([sample({"Other": 500})])["triggered"]


def test_absent_cid_resources_do_not_force_ocr():
    assert not classifier._get_cid_font_usage_signal_from_samples([sample()], {})["triggered"]


def test_generated_characters_do_not_inflate_font_support():
    value = sample({"ReadableCID": 100}, errors={"ReadableCID": 10})
    value["non_generated_char_count"] = 20
    value["font_non_generated_char_counts"] = {"ReadableCID": 20}
    assert not signal([value])["triggered"]


def test_pdfium_sample_records_font_decoding_quality():
    with pdfium.PdfDocument(text_pdf()) as document:
        value = classifier._collect_pdfium_text_samples(document, [0])[0]
    assert value["font_non_generated_char_counts"]["Helvetica"] >= 30
    assert value["font_non_generated_unicode_map_error_counts"].get("Helvetica", 0) == 0
    assert value["font_non_generated_abnormal_char_counts"].get("Helvetica", 0) == 0


def test_auto_route_keeps_readable_text_despite_missing_font_map(monkeypatch):
    # Inject only the font-resource risk; extraction and classification remain real.
    monkeypatch.setattr(classifier, "_get_font_resource_signals_pypdf", lambda *args: {
        "cid_without_to_unicode": {"triggered": True, "page_fonts": {0: {"Helvetica"}}},
        "latin_charset_with_to_unicode": {},
    })
    assert classifier.classify(text_pdf()) == "txt"


def test_auto_route_preserves_ocr_for_local_cid_decoding_errors(monkeypatch):
    value = sample({"ReadableCID": 100, "Other": 4900}, errors={"ReadableCID": 10})
    value.update(text="Readable text " * 400, cleaned_text="Readabletext" * 400,
                 null_char_count=0, replacement_char_count=0, control_char_count=0,
                 private_use_char_count=0, unicode_map_error_count=10)
    monkeypatch.setattr(classifier, "_collect_pdfium_text_samples", lambda *args: [value])
    monkeypatch.setattr(classifier, "_get_font_resource_signals_pypdf", lambda *args: {
        "cid_without_to_unicode": {"triggered": True, "page_fonts": {0: {"ReadableCID"}}},
        "latin_charset_with_to_unicode": {},
    })
    assert classifier.classify(text_pdf()) == "ocr"


def test_scanned_page_without_text_remains_ocr():
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 500))
    canvas.rect(20, 20, 360, 460, fill=1)
    canvas.showPage()
    canvas.save()
    assert classifier.classify(stream.getvalue()) == "ocr"
