"""原生 PDF 的 NUL ToUnicode 回退：字体资源身份、原几何与两条读取路径。"""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument


def test_original_embedded_biomni_fonts_recover_both_symbols() -> None:
    """保留原件嵌入字体及损坏 CMap，在独立的合成页面中锁定两个真实错误。"""
    root = Path(__file__).parents[2]
    provenance = json.loads((root / "tests/fixtures/pdf_unicode_fallback_source.json").read_text(encoding="utf-8"))
    data = (root / provenance["fixture"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == provenance["fixture_sha256"]
    with PDFDocument(data) as document:
        for index, item in enumerate(provenance["pages"]):
            with patch.object(document, "_resolve_page_unicode", return_value={}):
                original = "".join(char["char"] for char in document[index].get_chars())
            assert "\0" in original
            assert "".join(char["char"] for char in document[index].get_chars()) == item["expected"]
            assert "".join(char["char"] for char in document[index].get_snapshot().text_geometry.chars) == item["expected"]


def _stream(data: bytes) -> DecodedStreamObject:
    """构造独立流对象，不依赖外部 PDF 文件或字体下载。"""
    stream = DecodedStreamObject()
    stream.set_data(data)
    return stream


def _font(writer: PdfWriter, mapping: bytes, names: tuple[str, ...] = ("beta", "similar", "greaterequal")):
    """让同名 Type1 字体具有明确的原字符码、glyph name 和故意损坏的 ToUnicode。"""
    cmap = b"begincmap\n1 begincodespacerange\n<00><FF>\nendcodespacerange\n" + mapping + b"\nendcmap"
    return writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Symbol"),
                NameObject("/Encoding"): DictionaryObject(
                    {
                        NameObject("/Differences"): ArrayObject(
                            [NumberObject(33), *(NameObject("/" + name) for name in names)]
                        ),
                    }
                ),
                NameObject("/FirstChar"): NumberObject(33),
                NameObject("/LastChar"): NumberObject(32 + len(names)),
                NameObject("/Widths"): ArrayObject([NumberObject(600) for _ in names]),
                NameObject("/ToUnicode"): writer._add_object(_stream(cmap)),
            }
        )
    )


def _pdf(
    mapping: bytes = b"2 beginbfrange\n<21><21><0000>\n<23><23><0000>\nendbfrange",
    *,
    names: tuple[str, ...] = ("beta", "similar", "greaterequal"),
    codes: bytes = b"2321",
    rotation: int = 0,
) -> bytes:
    """复现 Biomni 两个单字节 NUL 映射，正文内容与字体名称不参与修复判断。"""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): _font(writer, mapping, names)}),
        }
    )
    page[NameObject("/Contents")] = writer._add_object(_stream(b"BT /F1 14 Tf 1 0 0 1 30 120 Tm <" + codes + b"> Tj ET"))
    page.rotate(rotation)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("snapshot", [False, True])
def test_null_tounicode_recovers_named_symbols(snapshot: bool) -> None:
    """普通字符和紧凑快照都恢复 ≥、β，且修复不写回源 PDF。"""
    source = _pdf()
    with PDFDocument(source) as document:
        chars = document[0].get_snapshot().text_geometry.chars if snapshot else document[0].get_chars()
        assert "".join(char["char"] for char in chars) == "≥β"
        assert [char["source_indices"] for char in chars] == [(0,), (1,)]
        assert document.bytes == source


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_fallback_keeps_geometry_and_source_members(rotation: int) -> None:
    """独立改变页面旋转，恢复后只有字符和原 Unicode 值变化，未移动或复制源成员。"""
    source = _pdf(rotation=rotation)
    with PDFDocument(source) as document:
        with patch.object(document, "_resolve_page_unicode", return_value={}):
            original = document[0].get_chars_with_geometry()
        repaired = document[0].get_chars_with_geometry()
        snapshot = document[0].get_snapshot().text_geometry
    assert json.dumps(repaired.chars, default=list, sort_keys=True) == json.dumps(snapshot.chars, default=list, sort_keys=True)
    assert repaired.tight_bboxes == original.tight_bboxes
    assert repaired.origins == original.origins
    assert repaired.loose_bboxes == original.loose_bboxes
    for before, after in zip(original.chars, repaired.chars, strict=True):
        assert before["char"] == "\0"
        assert after["char"] in "≥β"
        for key in set(before) - {"char", "raw_code"}:
            assert list(before[key]) == list(after[key]) if key == "bbox" else before[key] == after[key]


@pytest.mark.parametrize(
    "mapping",
    [
        b"2 beginbfchar\n<21><0000>\n<23><0000>\nendbfchar",
        b"1 beginbfrange\n<21><23>[<0000><0042><0000>]\nendbfrange",
        b"% 9 beginbfchar <21><0058> endbfchar\n2 beginbfchar\n<21> <00 00> % <23><0058>\n<23><0000>\nendbfchar",
    ],
)
def test_cmap_syntax_does_not_change_font_identity(mapping: bytes) -> None:
    """字符、范围数组、空白和注释三类合法写法均按原始字符码恢复。"""
    with PDFDocument(_pdf(mapping)) as document:
        assert "".join(char["char"] for char in document[0].get_chars()) == "≥β"


@pytest.mark.parametrize(
    ("mapping", "names", "expected"),
    [
        (b"2 beginbfchar <21><0058> <23><2265> endbfchar", ("beta", "similar", "greaterequal"), "≥X"),
        (b"2 beginbfchar <21><0000> <23><2265> endbfchar", ("notARealGlyph", "similar", "greaterequal"), "≥\0"),
        (b"2 beginbfchar <21><0000> <23><2265> endbfchar", ("f_f_i", "similar", "greaterequal"), "≥\0"),
        (b"2 beginbfchar <21><0002> <23><2265> endbfchar", ("beta", "similar", "greaterequal"), "≥\x02"),
        (b"2 beginbfchar <21><0000> <23><2265> endbfchar", ("uniE123", "similar", "greaterequal"), "≥\0"),
        (b"2 beginbfchar <21><0000> <23><2265> endbfchar", ("uniFFFD", "similar", "greaterequal"), "≥\0"),
    ],
)
def test_valid_control_and_unknown_mappings_are_preserved(mapping: bytes, names: tuple[str, ...], expected: str) -> None:
    """有效 ToUnicode 优先；U+0002、未知名称、私用字和多字符 ligature 均不扩大修复。"""
    with PDFDocument(_pdf(mapping, names=names)) as document:
        assert "".join(char["char"] for char in document[0].get_chars()) == expected
        assert "".join(char["char"] for char in document[0].get_snapshot().text_geometry.chars) == expected


def test_normal_page_never_reads_font_resources() -> None:
    """正常页面不创建 pypdf reader，保持普通字符和紧凑快照的快速路径。"""
    with PDFDocument(_pdf(b"2 beginbfchar <21><03b2> <23><2265> endbfchar")) as document:
        with patch.object(document, "_get_form_resource_page", side_effect=AssertionError("unnecessary resources")):
            assert "".join(char["char"] for char in document[0].get_chars()) == "≥β"
            assert "".join(char["char"] for char in document[0].get_snapshot().text_geometry.chars) == "≥β"
        assert not document._unicode_fallbacks


def test_recovery_is_cached_without_handles_and_close_keeps_snapshot() -> None:
    """跨普通读取与快照复用整数映射，关闭文档清理缓存后快照仍完整。"""
    with PDFDocument(_pdf()) as document:
        with patch.object(document, "_get_form_resource_page", wraps=document._get_form_resource_page) as read:
            document[0].get_chars()
            snapshot = document[0].get_snapshot()
            assert read.call_count == 1
        assert document._unicode_fallbacks == {0: {0: ord("≥"), 1: ord("β")}}
    assert not document._unicode_fallbacks
    assert "".join(char["char"] for char in snapshot.text_geometry.chars) == "≥β"


@pytest.mark.parametrize("form", [False, True])
def test_same_name_and_shared_cmap_keep_per_font_differences(form: bool) -> None:
    """同名字体甚至共享 ToUnicode 时仍按各自 Differences 恢复，覆盖 Form 内资源同名遮蔽。"""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    first = _font(writer, b"1 beginbfchar <21><0000> endbfchar", ("beta",))
    second = _font(writer, b"1 beginbfchar <21><0000> endbfchar", ("greaterequal",))
    second.get_object()[NameObject("/ToUnicode")] = first.get_object().raw_get("/ToUnicode")
    fonts = DictionaryObject({NameObject("/F1"): first, NameObject("/F2"): second})
    resources = DictionaryObject({NameObject("/Font"): fonts})
    if form:
        child = _stream(b"BT /F1 14 Tf 1 0 0 1 0 0 Tm <21> Tj ET")
        child.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): ArrayObject([NumberObject(v) for v in [0, -10, 30, 30]]),
                NameObject("/Resources"): DictionaryObject(
                    {NameObject("/Font"): DictionaryObject({NameObject("/F1"): second})}
                ),
            }
        )
        # 两级 Form 都复用相同资源名，内容流 q/Q 和 cm 决定实例位置。
        parent = _stream(b"/Child Do")
        parent.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): child["/BBox"],
                NameObject("/Resources"): DictionaryObject(
                    {NameObject("/XObject"): DictionaryObject({NameObject("/Child"): writer._add_object(child)})}
                ),
            }
        )
        resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Form"): writer._add_object(parent)})
        content = b"BT /F1 14 Tf 1 0 0 1 30 120 Tm <21> Tj ET q 1 0 0 1 38.4 120 cm /Form Do Q"
    else:
        content = b"BT /F1 14 Tf 1 0 0 1 30 120 Tm <21> Tj /F2 14 Tf <21> Tj ET"
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = writer._add_object(_stream(content))
    output = BytesIO()
    writer.write(output)
    source = output.getvalue()
    with PDFDocument(source) as document:
        chars = document[0].get_chars()
        assert "".join("".join(char["char"] for char in chars).split()) == "β≥"
        assert "".join("".join(char["char"] for char in document[0].get_snapshot().text_geometry.chars).split()) == "β≥"
        assert document.bytes == source
        # 原 reader 字典仍共享损坏的流；临时视图的修复没有污染其它消费者。
        original_fonts = document._get_form_resource_page(0)["/Resources"]["/Font"]
        assert b"<0000>" in original_fonts["/F1"]["/ToUnicode"].get_data()
        assert b"<0000>" in original_fonts["/F2"]["/ToUnicode"].get_data()


def test_unsupported_cmap_keeps_original_text() -> None:
    """无法完整解释的继承 CMap 保持未恢复，不丢字也不覆盖原提取器结果。"""
    mapping = b"/UnresolvedParent usecmap\n2 beginbfchar <21><0000> <23><0000> endbfchar"
    with PDFDocument(_pdf(mapping)) as document:
        assert "".join(char["char"] for char in document[0].get_chars()) == "\0\0"


@pytest.mark.parametrize("snapshot", [False, True])
def test_invalid_recovery_contract_propagates(snapshot: bool) -> None:
    """绑定层拒绝越界或改写正常位置的恢复结果，程序错误不能伪装成识别失败。"""
    with PDFDocument(_pdf()) as document:
        with patch.object(document, "_resolve_page_unicode", return_value={999: ord("β")}):
            with pytest.raises(ValueError, match="Unicode fallback"):
                document[0].get_snapshot() if snapshot else document[0].get_chars()
