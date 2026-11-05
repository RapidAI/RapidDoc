"""原生表格区域与单元格共用的保守文本质量检查。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ....._compute_backend import get_native
from .._native_geometry import raw_bbox
from ..spatial_text import _normalize_table_text
from .contracts import NativeTableInput
from .geometry import normalize_angle, normalize_bbox, bbox_center, page_bbox_to_table_local


@dataclass(frozen=True, slots=True)
class TableTextQuality:
    """保存不含原文的质量统计，准入与准出使用完全相同的拒绝阈值。"""

    stage: Literal["input", "output"]
    text_chars: int
    abnormal_chars: int

    @property
    def abnormal_ratio(self) -> float:
        """有效字符包含异常字符，正常空白不参与比例计算。"""
        return self.abnormal_chars / self.text_chars if self.text_chars else 0.0

    @property
    def rejected(self) -> bool:
        """至少两个异常且占比不低于百分之五时拒绝原生结果。"""
        return self.abnormal_chars >= 2 and self.abnormal_chars * 20 >= self.text_chars

    def diagnostic(self) -> dict[str, str | int | float | bool]:
        """提供稳定诊断，避免把乱码原文写进日志或用户协议。"""
        return {
            "stage": self.stage,
            "text_chars": self.text_chars,
            "abnormal_chars": self.abnormal_chars,
            "abnormal_ratio": self.abnormal_ratio,
            "rejected": self.rejected,
        }


def measure_table_text_quality(parts: Iterable[str], *, stage: Literal["input", "output"]) -> TableTextQuality:
    """先执行现有表格规范化，再统计控制字符、替代字符与孤立代理项。"""
    total = abnormal = 0
    for part in parts:
        text = _normalize_table_text(part)
        index = 0
        while index < len(text):
            char = text[index]
            code = ord(char)
            index += 1
            if 0xD800 <= code <= 0xDBFF and index < len(text) and 0xDC00 <= ord(text[index]) <= 0xDFFF:
                # 有效 UTF-16 代理对等价于一个正常非 BMP 字符，不能当成两个孤立代理项。
                index += 1
                total += 1
                continue
            bad = (code < 32 and char not in "\t\r\n") or 127 <= code <= 159 or code == 0xFFFD or 0xD800 <= code <= 0xDFFF
            # 必须先识别非法控制字符，避免 str.isspace() 把部分乱码控制码静默吞掉。
            if bad or not char.isspace():
                total += 1
                abnormal += int(bad)
    return TableTextQuality(stage, total, abnormal)


def measure_table_input_quality(table_input: NativeTableInput) -> TableTextQuality:
    """复用表格区域的中心点筛选，排除表外及 PDFium 生成的字符。"""
    bbox = normalize_bbox(table_input.table_bbox)
    if bbox is None:
        return TableTextQuality("input", 0, 0)
    native = get_native()
    if native is None:
        selected = []
        for index, char in enumerate(table_input.chars):
            absolute = normalize_bbox(char.get("bbox"))
            if absolute is None:
                continue
            x, y = bbox_center(absolute)
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                selected.append((index, absolute, page_bbox_to_table_local(absolute, bbox, normalize_angle(table_input.angle))))
    else:
        selected = native.table_boxes(
            [raw_bbox(char.get("bbox")) for char in table_input.chars],
            bbox,
            normalize_angle(table_input.angle),
            normalize_bbox,
        )
    return measure_table_text_quality(
        (
            str(table_input.chars[index].get("char") or "")
            for index, _absolute, local in selected
            if local is not None and not table_input.chars[index].get("is_generated", False)
        ),
        stage="input",
    )


__all__ = ["TableTextQuality", "measure_table_input_quality", "measure_table_text_quality"]
