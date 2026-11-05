"""BIFF 尾部检查的零拷贝与既有边界语义回归。"""

from __future__ import annotations

import struct
from unittest.mock import Mock

import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.office.errors import LegacyOfficeResourceLimitError
from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls import records as records_module
from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.records import EOF, RecordBudget, iter_records


class _SliceTrackingBytes(bytes):
    """记录没有显式终点的切片，识别逐条复制剩余工作簿的回退。"""

    suffix_slices: int = 0

    def __getitem__(self, key: int | slice) -> int | bytes:
        """保留原字节语义，只累计可能复制整个尾部的切片。"""
        if isinstance(key, slice) and key.stop is None:
            self.suffix_slices += 1
        return super().__getitem__(key)


def test_xls_iteration_does_not_copy_remaining_stream_per_record() -> None:
    """验证全流扫描仅实体化记录载荷，不逐条复制尚未访问的全部字节。"""
    data = _SliceTrackingBytes(struct.pack("<HH6s", 0x0203, 6, b"123456") * 256 + b"\x00" * 4096)
    budget = RecordBudget()

    records = list(iter_records(data, budget=budget))

    assert len(records) == budget.count == 256
    assert [record.offset for record in records] == list(range(0, 2560, 10))
    assert all(record.payload == b"123456" and type(record.payload) is bytes for record in records)
    assert data.suffix_slices == 0


@pytest.mark.parametrize(
    ("data", "start", "expected_types", "warning_offset"),
    [
        (b"", 0, [], None),
        (b"", -1, [], -1),
        (b"\x00" * 8, 0, [], None),
        (b"\x00" * 8, -1, [], None),
        (b"\x01\x00", -1, [], None),
        (b"\x01\x00", -3, [], -3),
        (b"\x01", 0, [], 0),
        (b"\x01", -1, [], -1),
        (b"\x01", 2, [], None),
        (struct.pack("<HH", 1, 5) + b"x", 0, [], 0),
        (struct.pack("<HH", 1, 0) + b"\x00" * 4, 0, [1], None),
        (b"\x00" * 4 + struct.pack("<HH", 1, 0), 0, [0, 1], None),
        (struct.pack("<HH", 1, 0) + b"x", 0, [1], 4),
    ],
)
def test_xls_view_preserves_padding_and_invalid_offset_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    start: int,
    expected_types: list[int],
    warning_offset: int | None,
) -> None:
    """验证空流、负起点、零记录、零尾部和截断记录仍按原顺序告警或停止。"""
    warning = Mock()
    monkeypatch.setattr(records_module.logger, "warning", warning)
    budget = RecordBudget()

    records = list(iter_records(data, start=start, budget=budget))

    assert [record.record_type for record in records] == expected_types
    assert budget.count == len(expected_types)
    if warning_offset is None:
        warning.assert_not_called()
    else:
        warning.assert_called_once_with("XLS_TRUNCATED_RECORD: workbook stream ends mid-record at byte {}", warning_offset)


def test_xls_eof_stops_before_invalid_tail_and_before_extra_budget_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证显式 EOF 停止不会检查后续坏载荷或额外消耗记录预算。"""
    warning = Mock()
    monkeypatch.setattr(records_module.logger, "warning", warning)
    monkeypatch.setattr(records_module, "MAX_RECORDS", 1)
    data = struct.pack("<HH", EOF, 0) + b"x"
    budget = RecordBudget()

    assert [record.record_type for record in iter_records(data, stop_at_eof=True, budget=budget)] == [EOF]
    assert budget.count == 1
    warning.assert_not_called()


def test_xls_padding_does_not_hide_later_records_from_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证后方仍有非零记录时，零头记录继续计费并在原有边界抛资源错误。"""
    monkeypatch.setattr(records_module, "MAX_RECORDS", 1)
    data = b"\x00" * 4 + struct.pack("<HH", 1, 0)
    budget = RecordBudget()

    with pytest.raises(LegacyOfficeResourceLimitError, match="max_records=1"):
        list(iter_records(data, budget=budget))

    assert budget.count == 2


def test_xls_zero_tail_stops_without_exceeding_exact_record_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证恰好用尽预算的合法记录仍可在零填充处无错误结束。"""
    monkeypatch.setattr(records_module, "MAX_RECORDS", 1)
    data = struct.pack("<HH", 1, 0) + b"\x00" * 4096
    budget = RecordBudget()

    assert [record.record_type for record in iter_records(data, budget=budget)] == [1]
    assert budget.count == 1
