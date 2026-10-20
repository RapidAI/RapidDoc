"""超限 BLIP 仅丢弃图片，正文、后续图片引用及文档级预算保持有效。"""

from io import BytesIO
import struct
import zlib

import olefile
import pytest
from loguru import logger

from _legacy_doc_test_utils import build_doc
from _legacy_ppt_test_utils import _build_cfb, _officeart_record, build_sparse_notes_ppt
from _legacy_xls_test_utils import SheetFixture, biff_record, build_xls, label_cell
from _mtef_test_utils import _TINY_PNG

from rapid_doc._vendor.docvortex_core.analyzers.native import DocModel, PptModel, XlsModel
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc import images as doc_images
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.fib import FCLCB_DGG_INFO, FCLCB_SHAPE_MAIN
from rapid_doc._vendor.docvortex_core.analyzers.native.office.doc.records import DocBudget
from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeResourceLimitError
from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy import officeart
from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt import parser as ppt_parser
from rapid_doc._vendor.docvortex_core.analyzers.native.office.ppt.records import PptRecord, RecordBudget


def _metafile(*, declared_size=None, data=b"small metafile", compression=0, record_type=0xF01B, raw=False):
    payload = zlib.compress(data) if compression == 0 else data
    if raw:
        payload = payload[2:-4]
    header = bytearray(34)
    struct.pack_into("<I", header, 0, len(data) if declared_size is None else declared_size)
    struct.pack_into("<I", header, 28, len(payload))
    header[32] = compression
    return _officeart_record(record_type, bytes(16) + header + payload)


def _bse(blip):
    # 内嵌 BLIP，延迟偏移无效，确保第二张图片不会错误引用第一张。
    body = bytearray(36)
    struct.pack_into("<I", body, 28, 0xFFFFFFFF)
    return _officeart_record(0xF007, body + blip, version=2)


def _bstore(bad=None):
    bad = bad if bad is not None else _metafile(declared_size=officeart.MAX_ENTRY_BYTES + 1)
    good = _officeart_record(0xF01E, bytes(17) + _TINY_PNG)
    return _officeart_record(0xF001, _bse(bad) + _bse(good), version=15)


@pytest.fixture
def warnings():
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("kind", ["declared_emf", "declared_wmf", "expanded", "uncompressed", "bitmap"])
def test_shared_picture_store_skips_one_image_and_preserves_bse_numbers(kind, monkeypatch, warnings):
    monkeypatch.setattr(officeart, "MAX_ENTRY_BYTES", 128)
    if kind.startswith("declared"):
        bad = _metafile(declared_size=129, record_type=0xF01A if kind.endswith("emf") else 0xF01B)
    elif kind == "expanded":
        bad = _metafile(declared_size=1, data=b"x" * 512)
    elif kind == "uncompressed":
        bad = _metafile(declared_size=1, data=b"x" * 512, compression=0xFE)
    else:
        bad = _officeart_record(0xF01E, bytes(17) + b"x" * 512)
    images = officeart.decode_bstore(_bstore(bad))
    assert set(images) == {2}
    assert images[2].data == _TINY_PNG
    assert warnings and any("1" in message and "max_entry_bytes" in message for message in warnings)


def test_declared_oversize_is_rejected_before_decompression(monkeypatch):
    blip = _metafile(declared_size=officeart.MAX_ENTRY_BYTES + 1)
    def forbidden(*args):
        pytest.fail("Oversize declaration must not reach decompression")
    monkeypatch.setattr(officeart.zlib, "decompressobj", forbidden)
    with pytest.raises(LegacyOfficeResourceLimitError):
        officeart.decode_blip(officeart.record_at(blip, 0))


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("size", [127, 128, 129])
def test_metafile_actual_output_limit_is_inclusive(size, raw, monkeypatch):
    monkeypatch.setattr(officeart, "MAX_ENTRY_BYTES", 128)
    record = officeart.record_at(_metafile(declared_size=1, data=b"x" * size, raw=raw), 0)
    if size > 128:
        with pytest.raises(LegacyOfficeResourceLimitError, match="decompression"):
            officeart.decode_blip(record)
    else:
        assert officeart.decode_blip(record).data == b"x" * size


def test_doc_inline_oversize_does_not_try_magic_fallback(warnings):
    # 超限矢量图内出现 PNG magic，也不能被当成正常图片重新读入。
    blip = _metafile(declared_size=officeart.MAX_ENTRY_BYTES + 1, data=_TINY_PNG, compression=0xFE)
    picf = struct.pack("<IH", len(blip) + 6, 6) + blip
    store = doc_images.ImageStore()
    assert doc_images.inline_picture(picf, offset=0, store=store, budget=DocBudget()) is None
    assert store.total == 0
    assert warnings


def _doc_with_floating_pictures():
    text = "Body survives oversized picture\r"
    with olefile.OleFileIO(BytesIO(build_doc(text))) as ole:
        word = bytearray(ole.openstream("WordDocument").read())
        table = bytearray(ole.openstream("1Table").read())
    shape_plc = struct.pack("<3I", 0, 1, len(text))
    shape_plc += b"".join(struct.pack("<I", shape_id) + bytes(22) for shape_id in (1, 2))
    shapes = b""
    for shape_id in (1, 2):
        fsp = _officeart_record(0xF00A, struct.pack("<II", shape_id, 0), version=2)
        fopt = _officeart_record(0xF00B, struct.pack("<HI", 0x0104, shape_id), version=3, instance=1)
        shapes += _officeart_record(0xF004, fsp + fopt, version=15)
    drawing = _officeart_record(0xF000, _bstore(), version=15)
    drawing += b"\0" + _officeart_record(0xF002, shapes, version=15)
    pair_offset = 110  # build_doc 的 FIB fc/lcb 表起点。
    for index, data in ((FCLCB_SHAPE_MAIN, shape_plc), (FCLCB_DGG_INFO, drawing)):
        struct.pack_into("<II", word, pair_offset + index * 8, len(table), len(data))
        table += data
    return _build_cfb([("WordDocument", bytes(word)), ("1Table", bytes(table)), ("Data", b"")])


def test_doc_keeps_text_and_normal_floating_picture_after_oversize(warnings):
    pages = DocModel().predict(BytesIO(_doc_with_floating_pictures()))
    assert "Body survives oversized picture" in str(pages)
    assert sum(block["type"] == "image" for page in pages for block in page) == 1
    assert warnings


def test_xls_keeps_cells_when_one_shared_picture_is_oversize(warnings):
    records = label_cell(0, 0, "Cell survives oversized picture") + label_cell(1, 0, "Tail")
    payload = build_xls([SheetFixture("Data", records)], globals_records=biff_record(0x00EB, _bstore()))
    pages = XlsModel().predict(BytesIO(payload))
    assert "Cell survives oversized picture" in str(pages) and "Tail" in str(pages)
    assert warnings


def test_ppt_picture_map_skips_oversize_without_renumbering(warnings):
    document = PptRecord(0, 15, 0, 0x03E8, _bstore())
    images = ppt_parser._picture_map(document, b"", RecordBudget())
    assert set(images) == {2} and images[2].data == _TINY_PNG
    assert warnings


def _ppt_with_bstore():
    # 向真实 persist fixture 的 DocumentContainer 加图片，并修正 persist 偏移。
    with olefile.OleFileIO(BytesIO(build_sparse_notes_ppt())) as ole:
        stream = ole.openstream("PowerPoint Document").read()
        user = bytearray(ole.openstream("Current User").read())
    edit_offset = struct.unpack_from("<I", user, 16)[0]
    directory_offset = struct.unpack_from("<I", stream, edit_offset + 20)[0]
    prefix = bytearray(stream[:directory_offset])
    document_length = struct.unpack_from("<I", prefix, 4)[0]
    extra = _bstore()
    prefix[8:8] = extra
    struct.pack_into("<I", prefix, 4, document_length + len(extra))
    directory = bytearray(stream[directory_offset:edit_offset])
    # fixture 包含一条 persist run 及四个对象偏移。
    for offset in (12, 16, 20, 24):
        value = struct.unpack_from("<I", directory, offset)[0]
        struct.pack_into("<I", directory, offset, value + len(extra) if value else 0)
    edit_length = struct.unpack_from("<I", stream, edit_offset + 4)[0] + 8
    edit = bytearray(stream[edit_offset:edit_offset + edit_length])
    struct.pack_into("<I", edit, 20, directory_offset + len(extra))
    struct.pack_into("<I", user, 16, edit_offset + len(extra))
    return _build_cfb([("Current User", bytes(user)), ("PowerPoint Document", bytes(prefix + directory + edit))])


def test_ppt_keeps_slides_and_notes_after_oversize(warnings):
    pages = PptModel().predict(BytesIO(_ppt_with_bstore()))
    assert len(pages) == 2
    assert "First slide text" in str(pages[0])
    assert "Second slide text" in str(pages[1]) and "Notes for the second slide" in str(pages[1])
    assert warnings


@pytest.mark.parametrize("route", ["shared", "ppt", "doc_inline"])
def test_document_record_budget_still_fails_hard(route):
    def exhausted():
        raise LegacyOfficeResourceLimitError("document record budget exhausted")
    with pytest.raises(LegacyOfficeResourceLimitError, match="record budget exhausted"):
        if route == "shared":
            officeart.decode_bstore(_bstore(), charge=exhausted)
        elif route == "ppt":
            class ExhaustedBudget:
                charge = staticmethod(exhausted)
            ppt_parser._picture_map(PptRecord(0, 15, 0, 0x03E8, _bstore()), b"", ExhaustedBudget())
        else:
            class ExhaustedBudget:
                charge = staticmethod(exhausted)
            blip = _metafile()
            picf = struct.pack("<IH", len(blip) + 6, 6) + blip
            doc_images.inline_picture(picf, offset=0, store=doc_images.ImageStore(), budget=ExhaustedBudget())


@pytest.mark.parametrize("route", ["shared", "ppt", "doc_inline"])
def test_total_asset_budget_still_fails_hard(route, monkeypatch):
    good = _officeart_record(0xF01E, bytes(17) + _TINY_PNG)
    if route == "shared":
        monkeypatch.setattr(officeart, "MAX_ASSET_TOTAL_BYTES", 1)
        call = lambda: officeart.decode_bstore(_officeart_record(0xF001, _bse(good), version=15))
    elif route == "ppt":
        monkeypatch.setattr(ppt_parser, "MAX_ASSET_TOTAL_BYTES", 1)
        call = lambda: ppt_parser._picture_map(PptRecord(0, 15, 0, 0x03E8, _bse(good)), b"", RecordBudget())
    else:
        monkeypatch.setattr(doc_images, "MAX_ASSET_TOTAL_BYTES", 1)
        picf = struct.pack("<IH", len(good) + 6, 6) + good
        call = lambda: doc_images.inline_picture(picf, offset=0, store=doc_images.ImageStore(), budget=DocBudget())
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_asset_total_bytes"):
        call()
