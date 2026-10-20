"""为 Flash 原生文本生成 loose/tight/origin 协商后的 canonical 几何。"""

from __future__ import annotations

import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Sequence, TypeAlias

from ._native_geometry import raw_bbox as _PLAIN_RECORD_RAW_BOX_REFERENCE

from ....document.pdf._document import PDFPageTextGeometry
from ....schema import BBox
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_union_many,
    _clip_bbox,
    _clip_validated_bbox,
    _coerce_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _rotate_origin_to_upright,
)
from .models import _LineItem

X_RELIABLE_PAIR_MIN = 30
X_STRONG_MEDIAN_RATIO = 1.30
X_STRONG_RATIO_THRESHOLD = 1.35
X_STRONG_RATIO_SHARE = 0.30
X_STRONG_NEXT_TIGHT_OVERLAP_SHARE = 0.30
X_SIBLING_PAIR_MIN = 10
X_SIBLING_MEDIAN_RATIO = 1.15
X_SIBLING_NEXT_TIGHT_OVERLAP_SHARE = 0.50
X_SIBLING_P90_RATIO = 1.50

ANCHOR_MIN_TIGHT_HEIGHT_RATIO = 0.38
Y_MIN_ANCHOR_COUNT = 4
Y_MIN_GEOMETRY_COVERAGE = 0.80
Y_DOMINANT_ROW_SHARE = 0.80
Y_NEIGHBOR_CORE_INTRUSION = 0.25
Y_MIN_REPEATED_LINES = 3
Y_MIN_REPEATED_SHARE = 0.50
Y_HEALTHY_LOOSE_SAMPLE_MIN = 20
Y_DOCUMENT_RISK_P95_RATIO = 2.20
STYLE_INFLATION_LOOSE_FONT_RATIO = 1.50
STYLE_INFLATION_LOOSE_TIGHT_RATIO = 1.80
STYLE_INFLATION_MIN_LINE_COUNT = 3
STYLE_INFLATION_MIN_LINE_SHARE = 0.50
STYLE_INFLATION_MIN_PAGE_COUNT = 2
STYLE_INFLATION_MIN_SCALE = 4.0
STYLE_TIER_MIN_MEMBER_COUNT = 2
STYLE_TIER_MIN_MEMBER_SHARE = 0.15
STYLE_TIER_GAP_RATIO = 1.35

RunKey: TypeAlias = tuple[str, float, int, int, int, str]
XRunStyleKey: TypeAlias = tuple[str, float, int, int, int]
LineKey: TypeAlias = tuple[int, int]
CharKey: TypeAlias = tuple[int, int]
LooseTierSample: TypeAlias = tuple[float, float, float, float]


@dataclass(slots=True)
class CharLayoutGeometry:
    """保存一个实际需要修复的字符几何及分轴状态。"""

    source_bbox: BBox
    tight_bbox: BBox
    origin: tuple[float, float]
    layout_bbox: BBox
    ink_bbox: BBox
    baseline: float
    advance: float | None
    em_height: float
    x_state: Literal["healthy", "abnormal", "unknown"]
    y_state: Literal["healthy", "abnormal", "unknown"]
    confidence: float


@dataclass(slots=True)
class LineGeometryRepair:
    """保存一条 legacy line 的 canonical 修复和 shadow 诊断。"""

    source_bbox: BBox
    layout_bbox: BBox
    ink_bbox: BBox | None
    baseline: float | None
    em_height: float
    state: Literal["healthy", "repair_x", "trim_y", "repair_xy", "uncertain"] = "healthy"
    confidence: float = 1.0
    split_y_candidate: bool = False
    repaired_char_count: int = 0
    y_intrusion_ratio: float = 0.0
    run_key: RunKey | None = None


@dataclass(slots=True)
class DocumentGeometryPlan:
    """保存文档级 run 结论、局部字符修复和逐行 canonical 几何。"""

    char_repairs: dict[CharKey, CharLayoutGeometry] = field(default_factory=dict)
    line_repairs: dict[LineKey, LineGeometryRepair] = field(default_factory=dict)
    line_style_scales: dict[LineKey, float] = field(default_factory=dict)
    # 单页极端字体矩阵异常有重复字形和健康同族行证明，允许在容器认领前校正。
    local_metric_repair_sources: set[LineKey] = field(default_factory=set)
    line_ink_bboxes: dict[LineKey, BBox] = field(default_factory=dict)
    line_baselines: dict[LineKey, float] = field(default_factory=dict)
    style_inflated_runs: set[RunKey] = field(default_factory=set)
    run_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    document_style_anomaly: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为 review 脚本可序列化的稳定诊断。"""

        return {
            "document_style_anomaly": self.document_style_anomaly,
            "run_diagnostics": self.run_diagnostics,
            "char_repairs": [
                {
                    "page_index": page_index,
                    "char_idx": char_idx,
                    "source_bbox": list(repair.source_bbox),
                    "tight_bbox": list(repair.tight_bbox),
                    "origin": list(repair.origin),
                    "layout_bbox": list(repair.layout_bbox),
                    "advance": repair.advance,
                    "em_height": repair.em_height,
                    "x_state": repair.x_state,
                    "y_state": repair.y_state,
                    "confidence": repair.confidence,
                }
                for (page_index, char_idx), repair in sorted(self.char_repairs.items())
            ],
            "line_repairs": [
                {
                    "page_index": page_index,
                    "source_index": source_index,
                    "source_bbox": list(repair.source_bbox),
                    "layout_bbox": list(repair.layout_bbox),
                    "ink_bbox": list(repair.ink_bbox) if repair.ink_bbox is not None else None,
                    "baseline": repair.baseline,
                    "em_height": repair.em_height,
                    "state": repair.state,
                    "confidence": repair.confidence,
                    "split_y_candidate": repair.split_y_candidate,
                    "repaired_char_count": repair.repaired_char_count,
                    "y_intrusion_ratio": repair.y_intrusion_ratio,
                    "run_key": list(repair.run_key) if repair.run_key is not None else None,
                }
                for (page_index, source_index), repair in sorted(self.line_repairs.items())
                if repair.state != "healthy" or repair.split_y_candidate
            ],
        }


@dataclass(frozen=True, slots=True)
class _DocumentGeometryRisk:
    """区分需要完整字符几何的布局风险与仅需字号校准的样式风险。"""

    layout: bool = False
    style: bool = False

    @property
    def any(self) -> bool:
        """返回当前文档是否需要进入完整字符样本收集。"""

        return self.layout or self.style


@dataclass(slots=True)
class _CharSample:
    """保存分析阶段使用的局部字符样本。"""

    page_index: int
    line: _LineItem
    position: int
    char_idx: int
    text: str
    source_bbox: BBox
    tight_bbox: BBox
    origin: tuple[float, float]
    local_source_bbox: BBox
    local_tight_bbox: BBox
    local_origin: tuple[float, float]
    run_key: RunKey
    font_size: float
    is_anchor: bool = False


@dataclass(slots=True)
class _RunStats:
    """保存一个字体 run 的 origin-advance 与覆盖统计。"""

    key: RunKey
    pair_ratios: list[float] = field(default_factory=list)
    pair_overlaps: list[bool] = field(default_factory=list)
    advances: list[float] = field(default_factory=list)
    tight_left_bearings: list[float] = field(default_factory=list)
    samples: list[_CharSample] = field(default_factory=list)
    strong_x_bad: bool = False
    sibling_x_bad: bool = False
    style_y_bad: bool = False
    median_advance: float | None = None
    median_tight_left_bearing: float = 0.0


@dataclass(slots=True)
class _LineAnalysis:
    """保存 Y 轴 row 聚类和邻行侵入判定所需信息。"""

    key: LineKey
    line: _LineItem
    samples: list[_CharSample]
    anchors: list[_CharSample]
    dominant: list[_CharSample]
    baseline: float
    tight_core: BBox
    local_source_bbox: BBox
    explicit_all_source_bbox: BBox
    legacy_local_bbox: BBox
    run_key: RunKey
    geometry_coverage: float
    dominant_share: float
    split_y_candidate: bool
    neighbors: list[_LineAnalysis] = field(default_factory=list)
    intrusion_ratio: float = 0.0
    y_candidate: bool = False


def _quantile(values: list[float], fraction: float) -> float:
    """返回确定性的线性插值分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _line_loose_tier_offsets(
    samples: list[LooseTierSample],
    em_height: float,
) -> tuple[float, float] | None:
    """把归一化 loose 高度分档，并返回最大异常档回缩到次档后的 ascent/descent。"""

    if len(samples) < 2 * STYLE_TIER_MIN_MEMBER_COUNT or em_height <= 0:
        return None
    normalized = sorted(
        (
            (ascent + descent) / max(font_size, tight_height, 0.1),
            ascent,
            descent,
            tight_height,
            font_size,
        )
        for ascent, descent, tight_height, font_size in samples
    )
    tiers: list[list[tuple[float, float, float, float, float]]] = []
    for item in normalized:
        if not tiers or item[0] > STYLE_TIER_GAP_RATIO * statistics.median(member[0] for member in tiers[-1]):
            tiers.append([item])
        else:
            tiers[-1].append(item)
    if len(tiers) < 2:
        return None
    upper_tier = tiers[-1]
    reference_tier = tiers[-2]
    minimum_members = max(
        STYLE_TIER_MIN_MEMBER_COUNT,
        math.ceil(STYLE_TIER_MIN_MEMBER_SHARE * len(samples)),
    )
    if len(upper_tier) < minimum_members or len(reference_tier) < STYLE_TIER_MIN_MEMBER_COUNT:
        return None
    upper_loose_heights = [member[1] + member[2] for member in upper_tier]
    upper_font_sizes = [member[4] for member in upper_tier if member[4] > 0]
    upper_tight_heights = [member[3] for member in upper_tier]
    if statistics.median(upper_loose_heights) <= STYLE_INFLATION_LOOSE_FONT_RATIO * (
        statistics.median(upper_font_sizes) if upper_font_sizes else 0.0
    ) or statistics.median(upper_loose_heights) <= STYLE_INFLATION_LOOSE_TIGHT_RATIO * max(
        0.1, _quantile(upper_tight_heights, 0.75)
    ):
        return None
    reference_ascent_ratios = [member[1] / max(member[4], member[3], 0.1) for member in reference_tier]
    reference_descent_ratios = [member[2] / max(member[4], member[3], 0.1) for member in reference_tier]
    return (
        statistics.median(reference_ascent_ratios) * em_height,
        statistics.median(reference_descent_ratios) * em_height,
    )


def _coerce_origin(value: Any) -> tuple[float, float] | None:
    """把 side map 中的 origin 收敛为有限二维坐标。"""

    try:
        origin = (float(value[0]), float(value[1]))
    except (IndexError, TypeError, ValueError):
        return None
    return origin if all(math.isfinite(item) for item in origin) else None


@lru_cache(maxsize=512)
def _normalized_font_family(name: str) -> str:
    """移除 PDF 子集前缀并归一化字体族名称。"""

    value = re.sub(r"^[A-Z]{6}\+", "", name)
    return re.sub(r"[\s_-]+", "", value).casefold() or "<unknown>"


@lru_cache(maxsize=8192)
def _script_group(text: str) -> str:
    """把字符归入字体 run 使用的宽粒度文字类别。"""

    if text.isascii() and text.isalpha():
        return "latin"
    if text.isdigit():
        return "digit"
    codepoint = ord(text[0]) if text else 0
    if (
        0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return "cjk"
    category = unicodedata.category(text[0]) if text else "Cn"
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    return "other"


def _font_run_metadata(font):
    """按既有顺序转换字体字段，特殊 Python 值保留原异常行为。"""
    font_name = str(font.get("name") or "<unknown>")
    try:
        font_size = float(font.get("size") or 0.0)
    except (TypeError, ValueError):
        font_size = 0.0
    try:
        flags = int(font.get("flags") or 0)
    except (TypeError, ValueError):
        flags = 0
    try:
        weight = int(round(float(font.get("weight") or 0.0) / 100.0) * 100)
    except (TypeError, ValueError):
        weight = 0
    return font_name, font_size, flags, weight


class _NativeFontUnsupported(Exception):
    """表示字体含自定义转换，原生文档必须在转换之前选择参考路径。"""


class _ReadOnlyFontCache:
    """Rust 几何已准备完成后的单行只读字体缓存，不跨行保留可变字典。"""

    def __init__(self, *, native_only=False):
        """每行建立有界身份表；原生文档只在首次见到字体时校验转换类型。"""
        self.values = {}
        self.native_only = native_only

    def metadata(self, font):
        """普通字体首见时校验字段，特殊转换前清空身份表以保留回调行为。"""
        cached = self.values.get(id(font))
        if cached is not None and cached[0] is font:
            return cached[1]
        if type(font) is not dict or any(
            type(font.get(name)) not in (str, int, float, bool, type(None)) for name in ("name", "size", "flags", "weight")
        ):
            self.values.clear()
            if self.native_only:
                raise _NativeFontUnsupported
            return _font_run_metadata(font)
        result = _font_run_metadata(font)
        if len(self.values) >= 4096:
            self.values.pop(next(iter(self.values)))
        self.values[id(font)] = (font, result)
        return result


def _font_run_key(char: dict[str, Any], angle: int, text: str, metadata_cache=None) -> tuple[RunKey, float]:
    """构造 run key；调用内缓存以字段值作键，字体发生变化时不会复用旧元数据。"""
    font = char.get("font")
    if type(metadata_cache) is _ReadOnlyFontCache and (type(char) is not dict or type(font) not in (dict, type(None))):
        metadata_cache.values.clear()
    font = font or {}
    if type(metadata_cache) is _ReadOnlyFontCache:
        if type(char) is not dict:
            metadata_cache.values.clear()
        font_name, font_size, flags, weight = metadata_cache.metadata(font)
        return _cached_run_key(font_name, font_size, flags, weight, angle, _script_group(text)), font_size
    key = None
    if metadata_cache is not None and type(font) is dict:
        values = tuple(font.get(name) for name in ("name", "size", "flags", "weight"))
        if type(values[0]) in (str, type(None)) and all(
            type(value) in (str, int, float, bool, type(None)) for value in values[1:]
        ):
            key = values
    metadata = metadata_cache.get(key) if key is not None else None
    if metadata is None:
        metadata = _font_run_metadata(font)
        if key is not None:
            metadata_cache[key] = metadata
    font_name, font_size, flags, weight = metadata
    return _cached_run_key(
        font_name,
        font_size,
        flags,
        weight,
        angle,
        _script_group(text),
    ), font_size


@lru_cache(maxsize=4096)
def _cached_run_key(
    font_name: str,
    font_size: float,
    flags: int,
    weight: int,
    angle: int,
    script: str,
) -> RunKey:
    """缓存同字体样式反复出现的规范化 run key。"""

    return (
        _normalized_font_family(font_name),
        round(font_size * 4.0) / 4.0,
        flags,
        weight,
        angle,
        script,
    )


def _is_anchor_text(text: str) -> bool:
    """仅让字母、数字和 CJK 等完整字形参与主要基线统计。"""

    if not text or text.isspace() or not text.isprintable():
        return False
    category = unicodedata.category(text[0])
    return category.startswith("L") or category.startswith("N")


def _source_bbox(
    char: dict[str, Any],
    geometry: PDFPageTextGeometry,
    char_idx: int,
) -> BBox | None:
    """按提取契约读取 loose：零旋转使用 char bbox，旋转字符才接受 side-map。"""

    raw_bbox = _coerce_bbox(char.get("bbox"))
    try:
        rotation = float(char.get("rotation") or 0.0)
    except (TypeError, ValueError):
        rotation = math.nan
    if math.isfinite(rotation) and abs(rotation) <= 1e-9:
        return raw_bbox
    side_bbox = _coerce_bbox(geometry.loose_bboxes.get(char_idx))
    if side_bbox is None or raw_bbox is None:
        return side_bbox or raw_bbox
    tight_bbox = _coerce_bbox(geometry.tight_bboxes.get(char_idx))
    stable_width = max(
        raw_bbox[2] - raw_bbox[0],
        (tight_bbox[2] - tight_bbox[0]) if tight_bbox is not None else 0.0,
        0.1,
    )
    if side_bbox[2] - side_bbox[0] > 1.6 * stable_width:
        return raw_bbox
    return side_bbox


def _style_line_is_inflated(
    source_height: float,
    font_sizes: Sequence[float],
    tight_heights: Sequence[float],
) -> bool:
    """按统一阈值判断一行 loose 高度是否显著偏离字号与 tight 字形。"""

    if not tight_heights:
        return False
    style_scale = max(
        statistics.median(font_sizes) if font_sizes else 0.0,
        _quantile(tight_heights, 0.75),
    )
    if style_scale < STYLE_INFLATION_MIN_SCALE:
        return False
    tight_height = max(0.1, _quantile(tight_heights, 0.75))
    return (
        source_height > STYLE_INFLATION_LOOSE_FONT_RATIO * style_scale
        and source_height > STYLE_INFLATION_LOOSE_TIGHT_RATIO * tight_height
    )


_PLAIN_RECORD_ANCHOR_REFERENCE = _is_anchor_text


def _plain_source_records(line, geometry, anchors_only):
    """普通行只按源顺序打包一次，坐标验证交给 Rust，特殊读取保留旧路径。"""
    from ._native_geometry import raw_bbox

    if (
        type(line) is not _LineItem
        or type(geometry) is not PDFPageTextGeometry
        or type(line.chars) is not list
        or any(type(values) is not dict for values in (geometry.tight_bboxes, geometry.loose_bboxes, geometry.origins))
    ):
        return None
    from ...._compute_backend import get_native
    from ....document.pdf.text._contracts import Bbox

    native = get_native()
    if native is not None and _is_anchor_text is _PLAIN_RECORD_ANCHOR_REFERENCE and raw_bbox is _PLAIN_RECORD_RAW_BOX_REFERENCE:
        records = native.plain_source_records_owned(
            line.chars,
            geometry.tight_bboxes,
            geometry.loose_bboxes,
            geometry.origins,
            anchors_only,
            _is_anchor_text,
            Bbox,
        )
        if records is not None:
            return records
    records = []
    for position, char in enumerate(line.chars):
        if type(char) is not dict or type(char.get("char")) is not str or type(char.get("char_idx")) is not int:
            return None
        text = char["char"]
        if not _is_anchor_text(text) if anchors_only else not text or not text.isprintable() or text.isspace():
            continue
        rotation = char.get("rotation")
        if type(rotation) not in (float, type(None)):
            return None
        rotation = rotation or 0.0
        index = char["char_idx"]
        records.append(
            (
                position,
                raw_bbox(char.get("bbox")),
                raw_bbox(geometry.loose_bboxes.get(index)) if abs(rotation) > 1e-9 or not math.isfinite(rotation) else None,
                raw_bbox(geometry.tight_bboxes.get(index)),
                geometry.origins.get(index),
                rotation,
            )
        )
    return records


def _prepared_line_geometry(line, geometry, page_size, *, anchors_only):
    """一次筛选并打包原生字符几何，保持风险与 canonical 样本的源顺序。"""
    from ...._compute_backend import get_native
    from ._native_geometry import raw_bbox

    native = get_native()
    if (
        native is not None
        and type(page_size) in (tuple, list)
        and len(page_size) == 2
        and all(
            type(value) is float and math.isfinite(value) or type(value) is int and -(2**53) <= value <= 2**53
            for value in page_size
        )
        and type(line) is _LineItem
        and type(line.angle) is int
        and -(2**31) <= line.angle < 2**31
    ):
        records = _plain_source_records(line, geometry, anchors_only)
        if records is not None:
            prepared = native.source_rows_plain(records, page_size, line.angle)
            if prepared is not None:
                for position, values in prepared:
                    char = line.chars[position]
                    yield position, char["char"], char["char_idx"], char, values
                return
    plain = (
        native is not None
        and type(line) is _LineItem
        and type(line.chars) is list
        and type(geometry.tight_bboxes) is dict
        and type(geometry.origins) is dict
        and type(geometry.loose_bboxes) is dict
        and all(
            type(char) is dict
            and type(char.get("char")) is str
            and type(char.get("char_idx")) is int
            and type(char.get("rotation")) in (float, type(None))
            and (
                (origin := geometry.origins.get(char["char_idx"])) is None
                or type(origin) in (tuple, list)
                and len(origin) == 2
                and all(type(value) is float for value in origin)
            )
            for char in line.chars
        )
    )
    selected = []
    sources, sides, tights, origins, rotations = [], [], [], [], []
    for position, char in enumerate(line.chars):
        text = str(char.get("char") or "")
        index = char.get("char_idx")
        if not _is_anchor_text(text) if anchors_only else not text or not text.isprintable() or text.isspace():
            continue
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        selected.append((position, text, index, char))
        if plain:
            try:
                rotation = float(char.get("rotation") or 0.0)
            except (TypeError, ValueError):
                rotation = math.nan
            sources.append(raw_bbox(char.get("bbox")))
            sides.append(
                raw_bbox(geometry.loose_bboxes.get(index)) if not math.isfinite(rotation) or abs(rotation) > 1e-9 else None
            )
            tights.append(raw_bbox(geometry.tight_bboxes.get(index)))
            origins.append(_coerce_origin(geometry.origins.get(index)))
            rotations.append(rotation)
    if native is None:
        for position, text, index, char in selected:
            source = _clip_validated_bbox(_source_bbox(char, geometry, index), page_size)
            tight = _clip_validated_bbox(_coerce_bbox(geometry.tight_bboxes.get(index)), page_size)
            origin = _coerce_origin(geometry.origins.get(index))
            if source is not None and tight is not None and origin is not None:
                yield (
                    position,
                    text,
                    index,
                    char,
                    (
                        source,
                        tight,
                        origin,
                        _rotate_bbox_to_upright(source, page_size, line.angle),
                        _rotate_bbox_to_upright(tight, page_size, line.angle),
                        _rotate_origin_to_upright(origin, page_size, line.angle),
                    ),
                )
        return
    if not plain:
        for _position, _text, index, char in selected:
            try:
                rotation = float(char.get("rotation") or 0.0)
            except (TypeError, ValueError):
                rotation = math.nan
            sources.append(raw_bbox(char.get("bbox")))
            sides.append(
                raw_bbox(geometry.loose_bboxes.get(index)) if not math.isfinite(rotation) or abs(rotation) > 1e-9 else None
            )
            tights.append(raw_bbox(geometry.tight_bboxes.get(index)))
            origins.append(_coerce_origin(geometry.origins.get(index)))
            rotations.append(rotation)
    prepared = native.source_rows(sources, sides, tights, origins, rotations, page_size, line.angle, _coerce_bbox)
    for (position, text, index, char), values in zip(selected, prepared, strict=True):
        if values is not None:
            source, tight, origin, local_source, local_tight, local_origin = values
            yield position, text, index, char, (source, tight, tuple(origin), local_source, local_tight, tuple(local_origin))


def _anchor_pair_statistics_python(rows, positive_source):
    """保留风险筛查和完整样本原来的相邻锚点公式与比较次序。"""
    output = []
    for index, (current, following) in enumerate(zip(rows, rows[1:])):
        source, tight, origin, key, size = current
        next_source, next_tight, next_origin, next_key, next_size = following
        if key != next_key:
            continue
        height = max(tight[3] - tight[1], next_tight[3] - next_tight[1])
        if abs(origin[1] - next_origin[1]) > max(0.5, 0.25 * height):
            continue
        advance = next_origin[0] - origin[0]
        width = max(tight[2] - tight[0], next_tight[2] - next_tight[0])
        if not 0.1 < advance <= max(5.0 * max(size, 1.0), 8.0 * width):
            continue
        source_width = source[2] - source[0]
        if positive_source and source_width <= 0:
            continue
        ratio = source_width / advance
        overlap = source[2] - next_tight[0] >= 0.05 * max(next_tight[2] - next_tight[0], 0.1)
        output.append((index, advance, ratio, overlap))
    return output


def _anchor_pair_statistics(rows, positive_source=False):
    """同一批数值同时服务风险筛查与 canonical 样本，返回索引及统计量。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is not None and all(
        type(size) is float
        and type(key) is tuple
        and all(type(v) in (str, int, float, bool, type(None)) for v in key)
        and all(
            type(box) in (tuple, list) and len(box) == length and all(type(v) is float for v in box)
            for box, length in ((source, 4), (tight, 4), (origin, 2))
        )
        for source, tight, origin, key, size in rows
    ):
        run_ids = {}
        result = native.anchor_pairs(
            [(run_ids.setdefault(key, len(run_ids)), source, tight, origin, size) for source, tight, origin, key, size in rows],
            positive_source,
        )
        if result is not None:
            return result
    return _anchor_pair_statistics_python(rows, positive_source)


def _document_requires_full_geometry(lines_by_page, geometries, page_sizes, *, owned_geometry_inputs=None):
    """标准数值路径连续累积全文风险，自定义规则明确使用 Python 参考实现。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is None
        or any(globals()[name] != value for name, value in _RISK_CONSTANTS.items())
        or any(globals()[name] is not value for name, value in _RISK_FUNCTIONS.items())
    ):
        return _document_requires_full_geometry_python(lines_by_page, geometries, page_sizes)
    if owned_geometry_inputs is not None and (
        type(owned_geometry_inputs) is not list or len(owned_geometry_inputs) != len(lines_by_page)
    ):
        owned_geometry_inputs = None
    state = native.NativeGeometryRisk()
    run_table = native.NativeGeometryRuns() if hasattr(native, "NativeGeometryRuns") else None
    run_ids = {}
    for page_index, (lines, geometry, page_size) in enumerate(zip(lines_by_page, geometries, page_sizes, strict=True)):
        for line in lines:
            if (
                type(line) is not _LineItem
                or type(line.source_index) is not int
                or not -(2**63) <= line.source_index < 2**63
                or not (
                    type(line.effective_height) is float
                    and math.isfinite(line.effective_height)
                    or type(line.effective_height) is int
                    and -(2**53) <= line.effective_height <= 2**53
                )
            ):
                return _document_requires_full_geometry_python(lines_by_page, geometries, page_sizes)
            skip_y = bool(
                line.angle != 0 or line.formula_candidate_only or line.restored_inline_cluster or line.compact_formula_cluster
            )
            owned = owned_geometry_inputs[page_index] if owned_geometry_inputs is not None else None
            owned_indices = None
            if owned is not None and run_table is not None:
                identities = owned[1]
                if type(line.chars) is not list:
                    owned_indices = False
                else:
                    owned_indices = [identities.get(id(char)) for char in line.chars]
                    if any(index is None for index in owned_indices):
                        owned_indices = False
            if owned_indices is not False and owned is not None and run_table is not None:
                early = owned[0].add_line(
                    state,
                    run_table,
                    page_index,
                    line.source_index,
                    line.effective_height,
                    skip_y,
                    owned_indices,
                    page_size,
                    line.angle,
                )
            elif owned_geometry_inputs is not None:
                # 回退行与 owned 批次分用两个从 0 起的 run 编号空间，混入同一风险器会合并无关
                # run、拆散同一 run；启用 owned 通道的文档任一行失配即整体回到参考路径。
                return _document_requires_full_geometry_python(lines_by_page, geometries, page_sizes)
            else:
                font_metadata = _ReadOnlyFontCache()
                entries = []
                for _position, text, _char_idx, char, prepared in _prepared_line_geometry(
                    line, geometry, page_size, anchors_only=True
                ):
                    key, size = _font_run_key(char, line.angle, text, font_metadata)
                    entries.append((run_ids.setdefault(key, len(run_ids)), prepared[3], prepared[4], prepared[5], size))
                early = state.add_line(
                    page_index,
                    line.source_index,
                    line.effective_height,
                    skip_y,
                    entries,
                )
            if early is None:
                return _document_requires_full_geometry_python(lines_by_page, geometries, page_sizes)
            if early:
                return _DocumentGeometryRisk(layout=True)
    layout, style = state.finish()
    return _DocumentGeometryRisk(layout=layout, style=style)


def _document_requires_full_geometry_python(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
) -> _DocumentGeometryRisk:
    """流式识别布局与样式风险，避免健康文档保留整本字符样本。"""

    x_ratios: dict[RunKey, list[float]] = defaultdict(list)
    x_overlaps: dict[RunKey, int] = defaultdict(int)
    y_ratios: list[float] = []
    y_extreme_runs: Counter[RunKey] = Counter()
    style_line_counts: Counter[RunKey] = Counter()
    style_inflated_lines: dict[RunKey, set[LineKey]] = defaultdict(set)
    font_metadata = {}
    from ...._compute_backend import get_native

    native_fonts = get_native() is not None
    for page_index, (lines, geometry, page_size) in enumerate(
        zip(lines_by_page, geometries, page_sizes, strict=True),
    ):
        for line in lines:
            if native_fonts:
                font_metadata = _ReadOnlyFontCache()
            entries: list[tuple[int, str, BBox, BBox, tuple[float, float], RunKey, float]] = []
            for position, text, _char_idx, char, prepared in _prepared_line_geometry(
                line, geometry, page_size, anchors_only=True
            ):
                run_key, font_size = _font_run_key(char, line.angle, text, font_metadata)
                _source, _tight, _origin, local_source, local_tight, local_origin = prepared
                entries.append(
                    (
                        position,
                        text,
                        local_source,
                        local_tight,
                        local_origin,
                        run_key,
                        font_size,
                    )
                )
            if not entries:
                continue
            height_q75 = _quantile([entry[3][3] - entry[3][1] for entry in entries], 0.75)
            anchors = [entry for entry in entries if entry[3][3] - entry[3][1] >= ANCHOR_MIN_TIGHT_HEIGHT_RATIO * height_q75]
            for index, _advance, ratio, overlap in _anchor_pair_statistics(
                [(entry[2], entry[3], entry[4], entry[5], entry[6]) for entry in anchors]
            ):
                key = anchors[index][5]
                x_ratios[key].append(ratio)
                if overlap:
                    x_overlaps[key] += 1

            anchors_by_run: dict[RunKey, list[tuple[int, str, BBox, BBox, tuple[float, float], RunKey, float]]] = defaultdict(
                list
            )
            for entry in anchors:
                anchors_by_run[entry[5]].append(entry)
            for run_key, run_entries in anchors_by_run.items():
                style_line_counts[run_key] += 1
                if _style_line_is_inflated(
                    line.effective_height,
                    [entry[6] for entry in run_entries if entry[6] > 0],
                    [entry[3][3] - entry[3][1] for entry in run_entries],
                ):
                    style_inflated_lines[run_key].add(
                        (page_index, line.source_index),
                    )

            # 四锚点门槛只约束 Y 分析；短行仍须为文档级 X 统计贡献相邻字符对。
            if len(anchors) < Y_MIN_ANCHOR_COUNT:
                continue
            if line.angle != 0 or line.formula_candidate_only or line.restored_inline_cluster or line.compact_formula_cluster:
                continue
            from ._ordered_statistics import ordered_clusters

            tolerance = max(0.5, 0.25 * height_q75)
            groups = ordered_clusters([entry[4][1] for entry in anchors], tolerance, last_only=True)
            clusters = [[anchors[index] for index in group] for group in groups]
            supported = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
            if len(supported) >= 2:
                return _DocumentGeometryRisk(layout=True)
            dominant = max(clusters, key=len)
            if len(dominant) / len(anchors) < Y_DOMINANT_ROW_SHARE:
                continue
            source_union = _bbox_union_many([entry[2] for entry in dominant])
            tight_union = _bbox_union_many([entry[3] for entry in dominant])
            tight_height = tight_union[3] - tight_union[1]
            if tight_height <= 0:
                continue
            ratio = (source_union[3] - source_union[1]) / tight_height
            y_ratios.append(ratio)
            if ratio >= 3.0:
                y_extreme_runs[Counter(entry[5] for entry in dominant).most_common(1)[0][0]] += 1

    layout_risk = False
    for run_key, ratios in x_ratios.items():
        pair_count = len(ratios)
        if pair_count < X_RELIABLE_PAIR_MIN:
            continue
        if (
            statistics.median(ratios) >= X_STRONG_MEDIAN_RATIO
            and sum(value > X_STRONG_RATIO_THRESHOLD for value in ratios) / pair_count >= X_STRONG_RATIO_SHARE
            and x_overlaps[run_key] / pair_count >= X_STRONG_NEXT_TIGHT_OVERLAP_SHARE
        ):
            layout_risk = True
            break
    layout_risk = (
        layout_risk
        or bool(y_ratios)
        and (
            _quantile(y_ratios, 0.95) >= Y_DOCUMENT_RISK_P95_RATIO
            or any(count >= Y_MIN_REPEATED_LINES for count in y_extreme_runs.values())
        )
    )
    style_risk = any(
        len(inflated_lines) >= STYLE_INFLATION_MIN_LINE_COUNT
        and len(inflated_lines) / style_line_counts[run_key] >= STYLE_INFLATION_MIN_LINE_SHARE
        and len({page_index for page_index, _source_index in inflated_lines}) >= STYLE_INFLATION_MIN_PAGE_COUNT
        for run_key, inflated_lines in style_inflated_lines.items()
        if style_line_counts[run_key] >= STYLE_INFLATION_MIN_LINE_COUNT
    )
    return _DocumentGeometryRisk(
        layout=layout_risk,
        style=style_risk,
    )


def _collect_samples(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
) -> tuple[list[_CharSample], dict[LineKey, list[_CharSample]]]:
    """收集具有合法 loose/tight/origin 的可见字符样本。"""

    samples: list[_CharSample] = []
    by_line: dict[LineKey, list[_CharSample]] = defaultdict(list)
    font_metadata = {}
    from ...._compute_backend import get_native

    native_fonts = get_native() is not None
    for page_index, (lines, geometry, page_size) in enumerate(zip(lines_by_page, geometries, page_sizes, strict=True)):
        for line in lines:
            if native_fonts:
                font_metadata = _ReadOnlyFontCache()
            for position, text, char_idx, char, prepared in _prepared_line_geometry(
                line, geometry, page_size, anchors_only=False
            ):
                source_bbox, tight_bbox, origin, local_source, local_tight, local_origin = prepared
                run_key, font_size = _font_run_key(char, line.angle, text, font_metadata)
                sample = _CharSample(
                    page_index=page_index,
                    line=line,
                    position=position,
                    char_idx=char_idx,
                    text=text,
                    source_bbox=source_bbox,
                    tight_bbox=tight_bbox,
                    origin=origin,
                    local_source_bbox=local_source,
                    local_tight_bbox=local_tight,
                    local_origin=local_origin,
                    run_key=run_key,
                    font_size=font_size,
                )
                samples.append(sample)
                by_line[(page_index, line.source_index)].append(sample)
        # canonical sample 已持有所需 source bbox；后续表格、脚本和字符回填只读取
        # tight/origin，逐页释放 loose side-map 可限制长文档峰值内存。
        geometry.loose_bboxes.clear()

    for line_samples in by_line.values():
        anchor_heights = [
            sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in line_samples if _is_anchor_text(sample.text)
        ]
        height_q75 = _quantile(anchor_heights, 0.75)
        for sample in line_samples:
            tight_height = sample.local_tight_bbox[3] - sample.local_tight_bbox[1]
            sample.is_anchor = (
                _is_anchor_text(sample.text) and height_q75 > 0 and tight_height >= ANCHOR_MIN_TIGHT_HEIGHT_RATIO * height_q75
            )
    return samples, by_line


def _build_run_stats(samples, by_line):
    """标准内部样本只读取一次，由 Rust 连续生成完整 run 统计与同族传播结论。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is not None
        and all(globals()[name] == value for name, value in _RISK_CONSTANTS.items())
        and all(globals()[name] is value for name, value in _RISK_FUNCTIONS.items())
        and type(samples) is list
        and type(by_line) in (dict, defaultdict)
    ):
        keys = list({sample.run_key for sample in samples})
        result = native.build_geometry_runs(samples, by_line, keys, _CharSample, _RunStats)
        if result is not None:
            return result
    return _build_run_stats_python(samples, by_line)


def _build_run_stats_python(
    samples: list[_CharSample],
    by_line: dict[LineKey, list[_CharSample]],
) -> dict[RunKey, _RunStats]:
    """统计同 run 相邻 origin advance 与 loose 覆盖下一 tight 的比例。"""

    runs = {key: _RunStats(key=key) for key in {sample.run_key for sample in samples}}
    for sample in samples:
        run = runs[sample.run_key]
        run.samples.append(sample)
        run.tight_left_bearings.append(sample.local_tight_bbox[0] - sample.local_origin[0])

    for line_samples in by_line.values():
        anchors = [sample for sample in line_samples if sample.is_anchor]
        anchors.sort(key=lambda sample: sample.position)
        for index, advance, ratio, overlap in _anchor_pair_statistics(
            [
                (sample.local_source_bbox, sample.local_tight_bbox, sample.local_origin, sample.run_key, sample.font_size)
                for sample in anchors
            ],
            positive_source=True,
        ):
            run = runs[anchors[index].run_key]
            run.pair_ratios.append(ratio)
            run.pair_overlaps.append(overlap)
            run.advances.append(advance)

    for run in runs.values():
        run.median_advance = statistics.median(run.advances) if run.advances else None
        run.median_tight_left_bearing = statistics.median(run.tight_left_bearings) if run.tight_left_bearings else 0.0
        pair_count = len(run.pair_ratios)
        if pair_count < X_RELIABLE_PAIR_MIN:
            continue
        median_ratio = statistics.median(run.pair_ratios)
        large_share = sum(value > X_STRONG_RATIO_THRESHOLD for value in run.pair_ratios) / pair_count
        overlap_share = sum(run.pair_overlaps) / pair_count
        run.strong_x_bad = (
            median_ratio >= X_STRONG_MEDIAN_RATIO
            and large_share >= X_STRONG_RATIO_SHARE
            and overlap_share >= X_STRONG_NEXT_TIGHT_OVERLAP_SHARE
        )

    bad_families = {run.key[0] for run in runs.values() if run.strong_x_bad}
    for run in runs.values():
        if run.strong_x_bad or run.key[0] not in bad_families or len(run.pair_ratios) < X_SIBLING_PAIR_MIN:
            continue
        median_ratio = statistics.median(run.pair_ratios)
        overlap_share = sum(run.pair_overlaps) / len(run.pair_overlaps)
        run.sibling_x_bad = (
            median_ratio >= X_SIBLING_MEDIAN_RATIO and overlap_share >= X_SIBLING_NEXT_TIGHT_OVERLAP_SHARE
        ) or _quantile(run.pair_ratios, 0.9) >= X_SIBLING_P90_RATIO
    return runs


def _mark_style_inflated_runs(
    runs: dict[RunKey, _RunStats],
) -> set[RunKey]:
    """标记跨页重复出现且 loose 高度同时偏离字号与 tight 的字体 run。"""

    output: set[RunKey] = set()
    for run in runs.values():
        samples_by_line: dict[LineKey, list[_CharSample]] = defaultdict(list)
        for sample in run.samples:
            if sample.is_anchor:
                samples_by_line[(sample.page_index, sample.line.source_index)].append(sample)
        if len(samples_by_line) < STYLE_INFLATION_MIN_LINE_COUNT:
            continue
        inflated_lines: list[LineKey] = []
        for line_key, line_samples in samples_by_line.items():
            if _style_line_is_inflated(
                line_samples[0].line.effective_height,
                [sample.font_size for sample in line_samples if sample.font_size > 0],
                [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in line_samples],
            ):
                inflated_lines.append(line_key)
        if (
            len(inflated_lines) >= STYLE_INFLATION_MIN_LINE_COUNT
            and len(inflated_lines) / len(samples_by_line) >= STYLE_INFLATION_MIN_LINE_SHARE
            and len({page_index for page_index, _source_index in inflated_lines}) >= STYLE_INFLATION_MIN_PAGE_COUNT
        ):
            run.style_y_bad = True
            output.add(run.key)
    return output


def _document_uses_global_style_calibration(
    style_inflated_runs: set[RunKey],
) -> bool:
    """只要存在已通过跨页重复证据的异常 run，就启用全文统一字体尺度。"""

    return bool(style_inflated_runs)


def _apply_style_scale_repairs(
    plan: DocumentGeometryPlan,
    style_inflated_runs: set[RunKey],
    by_line: dict[LineKey, list[_CharSample]],
) -> None:
    """异常文档触发后统一写入逐行 canonical 字号，不改变公开来源 bbox。"""

    if not style_inflated_runs:
        return
    for line_key, line_samples in by_line.items():
        anchor_samples = [sample for sample in line_samples if sample.is_anchor]
        if not anchor_samples:
            continue
        dominant_run = Counter(sample.run_key for sample in anchor_samples).most_common(1)[0][0]
        style_samples = [sample for sample in anchor_samples if sample.run_key == dominant_run]
        font_sizes = [sample.font_size for sample in style_samples if sample.font_size > 0]
        tight_heights = [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in style_samples]
        style_scale = max(
            statistics.median(font_sizes) if font_sizes else 0.0,
            _quantile(tight_heights, 0.75),
            0.1,
        )
        if style_scale < STYLE_INFLATION_MIN_SCALE:
            continue
        plan.line_style_scales[line_key] = style_scale


def _line_uses_repaired_style_scale(
    line: _LineItem,
    style_inflated_runs: set[RunKey],
) -> bool:
    """按字体族、字号、flags 和方向把异常 run 证据投影到当前视觉行。"""

    if line.font_signature is None or line.em_height <= 0:
        return False
    family = _normalized_font_family(line.font_signature[0])
    style_size = round(line.em_height * 4.0) / 4.0
    flags = line.font_signature[1]
    return any(
        family == run_key[0] and abs(style_size - run_key[1]) <= 0.25 and flags == run_key[2] and line.angle == run_key[4]
        for run_key in style_inflated_runs
    )


def _restore_stable_legacy_source_bboxes(
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """全文异常成立后改用原始字符 bbox，隔离仅存在于 loose side-map 的扰动。"""

    for (page_index, _source_index), line_samples in by_line.items():
        page_size = page_sizes[page_index]
        for sample in line_samples:
            raw_bbox = _clip_validated_bbox(
                _coerce_bbox(sample.line.chars[sample.position].get("bbox")),
                page_size,
            )
            if raw_bbox is None:
                continue
            sample.source_bbox = raw_bbox
            sample.local_source_bbox = _rotate_bbox_to_upright(
                raw_bbox,
                page_size,
                sample.line.angle,
            )


def _next_compatible_sample(
    current: _CharSample,
    line_samples: list[_CharSample],
) -> _CharSample | None:
    """返回同 run、同基线且 origin 正向的下一可见字符。"""

    for candidate in line_samples:
        if candidate.position <= current.position or candidate.run_key != current.run_key:
            continue
        tight_height = max(
            current.local_tight_bbox[3] - current.local_tight_bbox[1],
            candidate.local_tight_bbox[3] - candidate.local_tight_bbox[1],
        )
        if abs(current.local_origin[1] - candidate.local_origin[1]) > max(0.5, 0.25 * tight_height):
            continue
        advance = candidate.local_origin[0] - current.local_origin[0]
        if advance > 0.1:
            return candidate
    return None


def _x_run_style_key(run_key: RunKey) -> XRunStyleKey:
    """移除文字类别，返回字体样式和方向一致的 X 修复分组键。"""

    return run_key[:5]


def _samples_share_baseline(first: _CharSample, second: _CharSample) -> bool:
    """按 tight 字形高度判断两个字符是否位于同一局部基线。"""

    tight_height = max(
        first.local_tight_bbox[3] - first.local_tight_bbox[1],
        second.local_tight_bbox[3] - second.local_tight_bbox[1],
    )
    return abs(first.local_origin[1] - second.local_origin[1]) <= max(0.5, 0.25 * tight_height)


def _next_style_compatible_sample(
    current: _CharSample,
    line_samples: list[_CharSample],
) -> _CharSample | None:
    """返回同字体样式、同基线且 origin 正向的下一可见字符。"""

    style_key = _x_run_style_key(current.run_key)
    for candidate in line_samples:
        if (
            candidate.position <= current.position
            or _x_run_style_key(candidate.run_key) != style_key
            or not _samples_share_baseline(current, candidate)
        ):
            continue
        if candidate.local_origin[0] - current.local_origin[0] > 0.1:
            return candidate
    return None


def _build_x_char_repair(
    sample: _CharSample,
    donor_run: _RunStats,
    next_sample: _CharSample | None,
    page_size: tuple[float, float],
    *,
    left_bearing: float,
    use_donor_advance: bool = False,
) -> CharLayoutGeometry | None:
    """用可靠 run 的 advance 为单个字符构造包含 tight 字形的 X-only 修复。"""

    advance = (
        donor_run.median_advance
        if use_donor_advance or next_sample is None
        else next_sample.local_origin[0] - sample.local_origin[0]
    )
    if advance is None or advance <= 0:
        return None
    left = min(sample.local_tight_bbox[0], sample.local_origin[0] + left_bearing)
    right = max(sample.local_tight_bbox[2], sample.local_origin[0] + advance)
    if next_sample is not None:
        right = min(right, max(sample.local_tight_bbox[2], next_sample.local_origin[0]))
    if right <= left:
        return None
    layout_bbox = _clip_bbox(
        _rotate_bbox_from_upright(
            (
                left,
                sample.local_source_bbox[1],
                right,
                sample.local_source_bbox[3],
            ),
            page_size,
            sample.line.angle,
        ),
        page_size,
    )
    if layout_bbox is None:
        return None
    return CharLayoutGeometry(
        source_bbox=sample.source_bbox,
        tight_bbox=sample.tight_bbox,
        origin=sample.origin,
        layout_bbox=layout_bbox,
        ink_bbox=sample.tight_bbox,
        baseline=sample.local_origin[1],
        advance=advance,
        em_height=max(
            sample.font_size,
            sample.local_tight_bbox[3] - sample.local_tight_bbox[1],
        ),
        x_state="abnormal",
        y_state="healthy",
        confidence=min(1.0, len(donor_run.pair_ratios) / X_RELIABLE_PAIR_MIN),
    )


def _x_repair_donors(
    runs: dict[RunKey, _RunStats],
) -> dict[XRunStyleKey, _RunStats]:
    """按强异常、pair 数和完整键稳定选择各字体样式的 X 修复 donor。"""

    donors: dict[XRunStyleKey, _RunStats] = {}
    candidates = [
        run
        for run in runs.values()
        if (run.strong_x_bad or run.sibling_x_bad) and run.median_advance is not None and run.median_advance > 0
    ]
    for run in sorted(
        candidates,
        key=lambda item: (
            not item.strong_x_bad,
            -len(item.pair_ratios),
            item.key,
        ),
    ):
        donors.setdefault(_x_run_style_key(run.key), run)
    return donors


def _has_adjacent_repaired_style_sample(
    sample: _CharSample,
    line_samples: list[_CharSample],
    page_index: int,
    plan: DocumentGeometryPlan,
) -> bool:
    """确认稀疏字符紧邻同字体样式、同基线且已完成 X 修复的字符。"""

    style_key = _x_run_style_key(sample.run_key)
    return any(
        abs(candidate.position - sample.position) == 1
        and _x_run_style_key(candidate.run_key) == style_key
        and _samples_share_baseline(sample, candidate)
        and (page_index, candidate.char_idx) in plan.char_repairs
        for candidate in line_samples
    )


def _sparse_x_repair_affects_layout(
    sample: _CharSample,
    line_samples: list[_CharSample],
    donor_run: _RunStats,
    page_size: tuple[float, float],
) -> bool:
    """仅接纳撑开行尾或遮蔽跨栏硬间隙的稀疏字符 X 异常。"""

    donor_advance = donor_run.median_advance
    if donor_advance is None or donor_advance <= 0:
        return False
    tight_width = max(
        0.1,
        sample.local_tight_bbox[2] - sample.local_tight_bbox[0],
    )
    canonical_width = max(donor_advance, tight_width)
    source_width = sample.local_source_bbox[2] - sample.local_source_bbox[0]
    canonical_right = max(
        sample.local_tight_bbox[2],
        sample.local_origin[0] + donor_advance,
    )
    local_page_width = page_size[1] if sample.line.angle in {90, 270} else page_size[0]
    forward_overhang = sample.local_source_bbox[2] - canonical_right
    if source_width <= X_STRONG_RATIO_THRESHOLD * canonical_width or forward_overhang < max(
        2.0 * canonical_width, 0.01 * local_page_width
    ):
        return False

    baseline_samples = sorted(
        (
            candidate
            for candidate in line_samples
            if candidate.position != sample.position and _samples_share_baseline(sample, candidate)
        ),
        key=lambda candidate: candidate.position,
    )
    following = next(
        (candidate for candidate in baseline_samples if candidate.position > sample.position),
        None,
    )
    if following is None:
        right_edge = max(candidate.local_source_bbox[2] for candidate in [sample, *baseline_samples])
        return abs(sample.local_source_bbox[2] - right_edge) <= 0.25

    median_tight_width = statistics.median(
        max(
            0.1,
            candidate.local_tight_bbox[2] - candidate.local_tight_bbox[0],
        )
        for candidate in [sample, *baseline_samples]
    )
    hard_gap_threshold = max(
        15.0,
        3.0 * median_tight_width,
        0.02 * local_page_width,
    )
    return (
        following.local_tight_bbox[0] - canonical_right >= hard_gap_threshold
        and sample.local_source_bbox[2] >= following.local_origin[0]
    )


def _repair_x_chars(
    plan: DocumentGeometryPlan,
    runs: dict[RunKey, _RunStats],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """对强异常 run 及其同样式稀疏邻接字符重建前进方向 cell。"""

    donors = _x_repair_donors(runs)
    for line_key, line_samples in by_line.items():
        page_index, _source_index = line_key
        page_size = page_sizes[page_index]
        for sample in line_samples:
            run = runs[sample.run_key]
            if not (run.strong_x_bad or run.sibling_x_bad):
                continue
            next_sample = _next_compatible_sample(sample, line_samples)
            repair = _build_x_char_repair(
                sample,
                run,
                next_sample,
                page_size,
                left_bearing=run.median_tight_left_bearing,
            )
            if repair is None:
                continue
            plan.char_repairs[(page_index, sample.char_idx)] = repair

        for sample in line_samples:
            char_key = (page_index, sample.char_idx)
            run = runs[sample.run_key]
            donor = donors.get(_x_run_style_key(sample.run_key))
            if (
                char_key in plan.char_repairs
                or donor is None
                or len(run.pair_ratios) >= X_SIBLING_PAIR_MIN
                or not _has_adjacent_repaired_style_sample(
                    sample,
                    line_samples,
                    page_index,
                    plan,
                )
                or not _sparse_x_repair_affects_layout(
                    sample,
                    line_samples,
                    donor,
                    page_size,
                )
            ):
                continue
            repair = _build_x_char_repair(
                sample,
                donor,
                _next_style_compatible_sample(sample, line_samples),
                page_size,
                left_bearing=sample.local_tight_bbox[0] - sample.local_origin[0],
                use_donor_advance=True,
            )
            if repair is not None:
                plan.char_repairs[char_key] = repair


def _baseline_clusters(samples: list[_CharSample]) -> tuple[list[list[_CharSample]], float]:
    """按 origin-v 将 anchor 聚为独立视觉基线。"""

    tight_heights = [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in samples]
    tolerance = max(0.5, 0.25 * _quantile(tight_heights, 0.75))
    from ._ordered_statistics import ordered_clusters

    groups = ordered_clusters([sample.local_origin[1] for sample in samples], tolerance)
    clusters = [[samples[index] for index in group] for group in groups]
    return clusters, tolerance


def _record_line_canonical_metrics(
    plan: DocumentGeometryPlan,
    by_line: dict[LineKey, list[_CharSample]],
) -> None:
    """为无需改写 bbox 的普通行保存 tight 字形并集和 dominant origin 基线。"""

    for line_key, line_samples in by_line.items():
        if not line_samples:
            continue
        line = line_samples[0].line
        plan.line_ink_bboxes[line_key] = _bbox_union_many(
            [sample.tight_bbox for sample in line_samples],
        )
        if line.formula_candidate_only or line.compact_formula_cluster:
            continue
        anchors = [sample for sample in line_samples if sample.is_anchor]
        if len(anchors) < 2:
            continue
        clusters, _tolerance = _baseline_clusters(anchors)
        dominant = max(
            clusters,
            key=lambda cluster: (
                sum(sample.local_tight_bbox[2] - sample.local_tight_bbox[0] for sample in cluster),
                len(cluster),
            ),
        )
        if len(dominant) / len(anchors) < 0.5:
            continue
        plan.line_baselines[line_key] = statistics.median(sample.local_origin[1] for sample in dominant)


def _analyze_lines(
    lines_by_page: list[list[_LineItem]],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> dict[LineKey, _LineAnalysis]:
    """分析 legacy line 的 dominant anchor row 与 split shadow 候选。"""

    analyses: dict[LineKey, _LineAnalysis] = {}
    for page_index, lines in enumerate(lines_by_page):
        page_size = page_sizes[page_index]
        for line in lines:
            if line.angle != 0 or line.formula_candidate_only or line.restored_inline_cluster or line.compact_formula_cluster:
                continue
            line_key = (page_index, line.source_index)
            line_samples = by_line.get(line_key, [])
            visible_count = sum(
                1
                for char in line.chars
                if str(char.get("char") or "").isprintable() and not str(char.get("char") or "").isspace()
            )
            anchors = [sample for sample in line_samples if sample.is_anchor]
            if visible_count == 0 or len(anchors) < Y_MIN_ANCHOR_COUNT:
                continue
            clusters, _tolerance = _baseline_clusters(anchors)
            dominant = max(
                clusters,
                key=lambda cluster: (
                    sum(item.local_tight_bbox[2] - item.local_tight_bbox[0] for item in cluster),
                    len(cluster),
                ),
            )
            dominant_share = len(dominant) / len(anchors)
            geometry_coverage = len(line_samples) / visible_count
            supported_clusters = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
            baseline = statistics.median(sample.local_origin[1] for sample in dominant)
            tight_core = _bbox_union_many([sample.local_tight_bbox for sample in dominant])
            source_envelope = _bbox_union_many([sample.local_source_bbox for sample in anchors])
            all_source_envelope = _bbox_union_many([sample.local_source_bbox for sample in line_samples])
            run_key = Counter(sample.run_key for sample in dominant).most_common(1)[0][0]
            analyses[line_key] = _LineAnalysis(
                key=line_key,
                line=line,
                samples=line_samples,
                anchors=anchors,
                dominant=dominant,
                baseline=baseline,
                tight_core=tight_core,
                local_source_bbox=source_envelope,
                explicit_all_source_bbox=all_source_envelope,
                legacy_local_bbox=_rotate_bbox_to_upright(line.bbox, page_size, line.angle),
                run_key=run_key,
                geometry_coverage=geometry_coverage,
                dominant_share=dominant_share,
                split_y_candidate=len(supported_clusters) >= 2,
            )
    return analyses


def _has_document_y_risk(by_line: dict[LineKey, list[_CharSample]]) -> bool:
    """用稳定单基线行的尾部分布实现正常文档 Y 分析快速否决。"""

    ratios: list[float] = []
    extreme_by_run: Counter[RunKey] = Counter()
    for line_samples in by_line.values():
        line = line_samples[0].line if line_samples else None
        if (
            line is None
            or line.angle != 0
            or line.formula_candidate_only
            or line.restored_inline_cluster
            or line.compact_formula_cluster
        ):
            continue
        anchors = [sample for sample in line_samples if sample.is_anchor]
        if len(anchors) < Y_MIN_ANCHOR_COUNT:
            continue
        clusters, _tolerance = _baseline_clusters(anchors)
        supported_clusters = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
        if len(supported_clusters) >= 2:
            return True
        dominant = max(clusters, key=len)
        if len(dominant) / len(anchors) < Y_DOMINANT_ROW_SHARE:
            continue
        source_union = _bbox_union_many([sample.local_source_bbox for sample in dominant])
        tight_union = _bbox_union_many([sample.local_tight_bbox for sample in dominant])
        tight_height = tight_union[3] - tight_union[1]
        if tight_height <= 0:
            continue
        ratio = (source_union[3] - source_union[1]) / tight_height
        ratios.append(ratio)
        if ratio >= 3.0:
            run_key = Counter(sample.run_key for sample in dominant).most_common(1)[0][0]
            extreme_by_run[run_key] += 1
    return bool(ratios) and (
        _quantile(ratios, 0.95) >= Y_DOCUMENT_RISK_P95_RATIO
        or any(count >= Y_MIN_REPEATED_LINES for count in extreme_by_run.values())
    )


def _same_lane(first: _LineAnalysis, second: _LineAnalysis) -> bool:
    """用水平覆盖和左边缘近似判断两行是否属于同一栏。"""

    height = max(
        first.tight_core[3] - first.tight_core[1],
        second.tight_core[3] - second.tight_core[1],
    )
    return _bbox_axis_overlap_ratio(first.local_source_bbox, second.local_source_bbox, axis="x") >= 0.35 or abs(
        first.local_source_bbox[0] - second.local_source_bbox[0]
    ) <= 2.0 * max(height, 1.0)


def _assign_neighbors(analyses: dict[LineKey, _LineAnalysis]) -> None:
    """为每条 eligible line 选择上下最近的同栏正文行。"""

    by_page: dict[int, list[_LineAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        by_page[analysis.key[0]].append(analysis)
    for page_analyses in by_page.values():
        from ...._compute_backend import get_native

        native = get_native()
        if native is not None and all(
            type(item) is _LineAnalysis
            and type(item.local_source_bbox) in (tuple, list)
            and len(item.local_source_bbox) == 4
            and all(type(v) is float for v in item.local_source_bbox)
            and type(item.tight_core) in (tuple, list)
            and len(item.tight_core) == 4
            and all(type(v) is float for v in item.tight_core)
            and type(item.baseline) is float
            for item in page_analyses
        ):
            neighbors = native.line_neighbors(
                [
                    (id(item), item.local_source_bbox, item.tight_core[3] - item.tight_core[1], item.baseline)
                    for item in page_analyses
                ]
            )
            if neighbors is not None:
                for item, pair in zip(page_analyses, neighbors, strict=True):
                    item.neighbors.extend(page_analyses[index] for index in pair if index is not None)
                continue
        for current in page_analyses:
            candidates = [
                other
                for other in page_analyses
                if other is not current
                and _same_lane(current, other)
                and abs(other.baseline - current.baseline) >= max(2.0, 0.6 * (current.tight_core[3] - current.tight_core[1]))
            ]
            above = [other for other in candidates if other.baseline < current.baseline]
            below = [other for other in candidates if other.baseline > current.baseline]
            if above:
                current.neighbors.append(max(above, key=lambda other: other.baseline))
            if below:
                current.neighbors.append(min(below, key=lambda other: other.baseline))


def _intrusion_ratio(source_bbox: BBox, tight_core: BBox) -> float:
    """计算 loose envelope 侵入邻行 tight core 的纵向比例。"""

    overlap = max(0.0, min(source_bbox[3], tight_core[3]) - max(source_bbox[1], tight_core[1]))
    height = tight_core[3] - tight_core[1]
    return overlap / height if height > 0 else 0.0


def _mark_y_candidates(analyses: dict[LineKey, _LineAnalysis]) -> set[RunKey]:
    """按局部侵入和文档级重复支持确认允许 trim_y 的 run。"""

    _assign_neighbors(analyses)
    eligible_by_run: dict[RunKey, list[_LineAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        if (
            len(analysis.anchors) < Y_MIN_ANCHOR_COUNT
            or analysis.geometry_coverage < Y_MIN_GEOMETRY_COVERAGE
            or analysis.dominant_share < Y_DOMINANT_ROW_SHARE
        ):
            continue
        analysis.intrusion_ratio = max(
            (_intrusion_ratio(analysis.local_source_bbox, neighbor.tight_core) for neighbor in analysis.neighbors),
            default=0.0,
        )
        analysis.y_candidate = analysis.intrusion_ratio >= Y_NEIGHBOR_CORE_INTRUSION
        eligible_by_run[analysis.run_key].append(analysis)

    confirmed: set[RunKey] = set()
    for run_key, members in eligible_by_run.items():
        candidates = [member for member in members if member.y_candidate]
        if len(candidates) >= Y_MIN_REPEATED_LINES and len(candidates) / len(members) >= Y_MIN_REPEATED_SHARE:
            confirmed.add(run_key)
    return confirmed


def _healthy_loose_offsets_by_run(
    analyses: dict[LineKey, _LineAnalysis],
    confirmed_runs: set[RunKey],
) -> dict[RunKey, tuple[list[float], list[float]]]:
    """单次遍历收集各确认 run 的健康 loose ascent/descent。"""

    values: dict[RunKey, tuple[list[float], list[float]]] = {run_key: ([], []) for run_key in confirmed_runs}
    for analysis in analyses.values():
        if analysis.run_key not in confirmed_runs or analysis.y_candidate:
            continue
        ascents, descents = values[analysis.run_key]
        for sample in analysis.dominant:
            ascents.append(max(0.0, sample.local_origin[1] - sample.local_source_bbox[1]))
            descents.append(max(0.0, sample.local_source_bbox[3] - sample.local_origin[1]))
    return values


def _repair_y_lines(
    plan: DocumentGeometryPlan,
    analyses: dict[LineKey, _LineAnalysis],
    confirmed_runs: set[RunKey],
    x_bad_runs: set[RunKey],
    page_sizes: list[tuple[float, float]],
) -> None:
    """为确认异常的单基线行生成 baseline 锚定的 Y envelope。"""

    healthy_offsets = _healthy_loose_offsets_by_run(analyses, confirmed_runs)
    for line_key, analysis in analyses.items():
        current = plan.line_repairs.get(line_key)
        source_bbox = analysis.line.source_bbox or analysis.line.bbox
        should_trim = (
            analysis.run_key in confirmed_runs
            and (analysis.y_candidate or analysis.run_key in x_bad_runs)
            and len(analysis.anchors) >= Y_MIN_ANCHOR_COUNT
            and analysis.geometry_coverage >= Y_MIN_GEOMETRY_COVERAGE
            and analysis.dominant_share >= Y_DOMINANT_ROW_SHARE
        )
        if current is None and not analysis.split_y_candidate and not should_trim:
            continue
        if current is None:
            current = LineGeometryRepair(
                source_bbox=source_bbox,
                layout_bbox=analysis.line.bbox,
                ink_bbox=_bbox_union_many([sample.tight_bbox for sample in analysis.samples]) if analysis.samples else None,
                baseline=analysis.baseline,
                em_height=analysis.line.effective_height,
                split_y_candidate=analysis.split_y_candidate,
                run_key=analysis.run_key,
            )
            plan.line_repairs[line_key] = current
        else:
            current.split_y_candidate = analysis.split_y_candidate
            current.baseline = analysis.baseline
            current.run_key = analysis.run_key
        if not should_trim:
            continue

        local_tight_boxes = [sample.local_tight_bbox for sample in analysis.samples]
        tight_union = _bbox_union_many(local_tight_boxes)
        font_sizes = [sample.font_size for sample in analysis.dominant if sample.font_size > 0]
        tight_heights = [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in analysis.dominant]
        em_height = max(
            statistics.median(font_sizes) if font_sizes else 0.0,
            _quantile(tight_heights, 0.75),
            0.1,
        )
        healthy_ascents, healthy_descents = healthy_offsets[analysis.run_key]
        healthy_loose_height = (
            statistics.median(healthy_ascents) + statistics.median(healthy_descents)
            if healthy_ascents and healthy_descents
            else math.inf
        )
        tier_offsets = _line_loose_tier_offsets(
            [
                (
                    max(
                        0.0,
                        sample.local_origin[1] - sample.local_source_bbox[1],
                    ),
                    max(
                        0.0,
                        sample.local_source_bbox[3] - sample.local_origin[1],
                    ),
                    sample.local_tight_bbox[3] - sample.local_tight_bbox[1],
                    sample.font_size,
                )
                for sample in analysis.dominant
                if sample.run_key == analysis.run_key
            ],
            em_height,
        )
        if tier_offsets is not None:
            ascent, descent = tier_offsets
        elif len(healthy_ascents) >= Y_HEALTHY_LOOSE_SAMPLE_MIN and healthy_loose_height <= 1.5 * em_height:
            ascent = statistics.median(healthy_ascents)
            descent = statistics.median(healthy_descents)
        else:
            ascent = _quantile(
                [max(0.0, sample.local_origin[1] - sample.local_tight_bbox[1]) for sample in analysis.dominant],
                0.9,
            )
            descent = _quantile(
                [max(0.0, sample.local_tight_bbox[3] - sample.local_origin[1]) for sample in analysis.dominant],
                0.9,
            )
            padding = max(0.5, 0.08 * em_height)
            ascent += padding
            descent += padding

        top = min(analysis.baseline - ascent, tight_union[1])
        bottom = max(analysis.baseline + descent, tight_union[3])
        for neighbor in analysis.neighbors:
            midpoint = (analysis.baseline + neighbor.baseline) / 2.0
            if neighbor.baseline < analysis.baseline:
                top = max(top, min(midpoint, tight_union[1]))
            else:
                bottom = min(bottom, max(midpoint, tight_union[3]))
        legacy_height = analysis.legacy_local_bbox[3] - analysis.legacy_local_bbox[1]
        provenance_tolerance = max(1.05, 0.05 * legacy_height)
        preserve_legacy_typography = (
            abs(analysis.explicit_all_source_bbox[1] - analysis.legacy_local_bbox[1]) > provenance_tolerance
            or abs(analysis.explicit_all_source_bbox[3] - analysis.legacy_local_bbox[3]) > provenance_tolerance
        )
        if preserve_legacy_typography:
            # explicit loose 与 legacy char bbox 来源不同时保留已经健康的 legacy envelope；
            # 该分支也让 review 扰动只验证 side-map 修复，不重写既有 line membership。
            top = analysis.legacy_local_bbox[1]
            bottom = analysis.legacy_local_bbox[3]
        page_size = page_sizes[line_key[0]]
        local_current = _rotate_bbox_to_upright(current.layout_bbox, page_size, analysis.line.angle)
        layout_bbox = _clip_bbox(
            _rotate_bbox_from_upright(
                (local_current[0], top, local_current[2], bottom),
                page_size,
                analysis.line.angle,
            ),
            page_size,
        )
        if layout_bbox is None:
            continue
        current.layout_bbox = layout_bbox
        current.ink_bbox = _bbox_union_many([sample.tight_bbox for sample in analysis.samples])
        current.em_height = analysis.line.effective_height if preserve_legacy_typography else max(0.1, bottom - top)
        current.state = "repair_xy" if current.state == "repair_x" else "trim_y"
        current.confidence = min(analysis.geometry_coverage, analysis.dominant_share)
        current.y_intrusion_ratio = analysis.intrusion_ratio


def _build_line_repairs_from_x(
    plan: DocumentGeometryPlan,
    lines_by_page: list[list[_LineItem]],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """把已修复字符并集投影为只改变 X 的 canonical line。"""

    for page_index, lines in enumerate(lines_by_page):
        page_size = page_sizes[page_index]
        for line in lines:
            line_key = (page_index, line.source_index)
            line_samples = by_line.get(line_key, [])
            repaired = [
                plan.char_repairs[(page_index, sample.char_idx)]
                for sample in line_samples
                if (page_index, sample.char_idx) in plan.char_repairs
            ]
            if not repaired:
                continue
            local_source = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
            explicit_source = _bbox_union_many([sample.local_source_bbox for sample in line_samples])
            if abs(explicit_source[0] - local_source[0]) > 0.25 or abs(explicit_source[2] - local_source[2]) > 0.25:
                # 旋转字符或 shadow 扰动的 explicit loose 与 legacy bbox 来源不同，
                # 首版保留 legacy X，避免把诊断 side-map 变化误写入公开输出。
                continue
            local_boxes = []
            for sample in line_samples:
                repair = plan.char_repairs.get((page_index, sample.char_idx))
                page_bbox = repair.layout_bbox if repair is not None else sample.source_bbox
                local_boxes.append(_rotate_bbox_to_upright(page_bbox, page_size, line.angle))
            local_union = _bbox_union_many(local_boxes)
            layout_bbox = _clip_bbox(
                _rotate_bbox_from_upright(
                    (local_union[0], local_source[1], local_union[2], local_source[3]),
                    page_size,
                    line.angle,
                ),
                page_size,
            )
            if layout_bbox is None:
                continue
            ink_bbox = _bbox_union_many([sample.tight_bbox for sample in line_samples]) if line_samples else None
            plan.line_repairs[line_key] = LineGeometryRepair(
                source_bbox=line.bbox,
                layout_bbox=layout_bbox,
                ink_bbox=ink_bbox,
                baseline=None,
                em_height=line.effective_height,
                state="repair_x",
                confidence=min(repair.confidence for repair in repaired),
                repaired_char_count=len(repaired),
            )


def _run_diagnostics(runs: dict[RunKey, _RunStats]) -> list[dict[str, Any]]:
    """生成不依赖字符对象的 run 级可序列化诊断。"""

    output = []
    for run in sorted(runs.values(), key=lambda item: item.key):
        pair_count = len(run.pair_ratios)
        output.append(
            {
                "run_key": list(run.key),
                "sample_count": len(run.samples),
                "reliable_pair_count": pair_count,
                "median_ratio": statistics.median(run.pair_ratios) if pair_count else None,
                "p90_ratio": _quantile(run.pair_ratios, 0.9) if pair_count else None,
                "ratio_gt_1_35_share": (
                    sum(value > X_STRONG_RATIO_THRESHOLD for value in run.pair_ratios) / pair_count if pair_count else 0.0
                ),
                "next_tight_overlap_share": sum(run.pair_overlaps) / pair_count if pair_count else 0.0,
                "strong_x_bad": run.strong_x_bad,
                "sibling_x_bad": run.sibling_x_bad,
                "style_y_bad": run.style_y_bad,
            }
        )
    return output


def _prepare_run_style(
    plan: DocumentGeometryPlan,
    samples: list[_CharSample],
    by_line: dict[LineKey, list[_CharSample]],
    *,
    restore_pages: list[tuple[float, float]] | None = None,
) -> tuple[dict[RunKey, _RunStats], set[RunKey], bool]:
    """复用一次原生样本读取，连续计算 run 与全文样式；自定义规则保留原调用顺序。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is not None
        and type(samples) is list
        and type(by_line) in (dict, defaultdict)
        and all(globals()[name] == value for name, value in _RISK_CONSTANTS.items())
        and all(globals()[name] is value for name, value in _RISK_FUNCTIONS.items())
        and all(globals()[name] is value for name, value in _STYLE_FUNCTIONS.items())
        and (restore_pages is None or type(restore_pages) is list)
    ):
        keys = list({sample.run_key for sample in samples})
        from ....document.pdf.text._contracts import Bbox as CharacterBbox

        result = native.build_geometry_style(
            samples, by_line, keys, _CharSample, _RunStats, _LineItem, restore_pages, CharacterBbox
        )
        if result is not None:
            runs, inflated, scales, restored = result
            plan.style_inflated_runs = inflated
            plan.document_style_anomaly = bool(inflated)
            plan.line_style_scales.update(scales)
            return runs, inflated, restored
    runs = _build_run_stats(samples, by_line)
    inflated = _mark_style_inflated_runs(runs)
    plan.style_inflated_runs = inflated
    plan.document_style_anomaly = _document_uses_global_style_calibration(inflated)
    _apply_style_scale_repairs(plan, inflated if plan.document_style_anomaly else set(), by_line)
    return runs, inflated, False


def _prepare_owned_document(lines_by_page, geometries, page_sizes, *, layout=False, with_metrics=False):
    """自有 Rust 文档连续完成样本、锚点、run 和样式，仅为后续布局规则物化一次字符。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is None
        or any(globals()[name] != value for name, value in _RISK_CONSTANTS.items())
        or any(globals()[name] is not value for name, value in _RISK_FUNCTIONS.items())
        or any(globals()[name] is not value for name, value in _STYLE_FUNCTIONS.items())
        or any(globals()[name] is not value for name, value in _STYLE_ONLY_FUNCTIONS.items())
    ):
        return None
    document = native.NativeStyleDocument(layout)
    run_ids, run_keys = {}, []
    for page_index, (lines, geometry, page_size) in enumerate(zip(lines_by_page, geometries, page_sizes, strict=True)):
        if (
            type(geometry) is not PDFPageTextGeometry
            or any(type(values) is not dict for values in (geometry.tight_bboxes, geometry.origins, geometry.loose_bboxes))
            or type(page_size) not in (tuple, list)
            or len(page_size) != 2
            or not all(type(v) is float and math.isfinite(v) or type(v) is int and -(2**53) <= v <= 2**53 for v in page_size)
        ):
            return None
        for line in lines:
            if (
                type(line) is not _LineItem
                or (
                    with_metrics
                    and any(
                        type(value) is not bool
                        for value in (line.formula_candidate_only, line.compact_formula_cluster, line.restored_inline_cluster)
                    )
                )
                or type(line.angle) is not int
                or not -(2**31) <= line.angle < 2**31
                or type(line.source_index) is not int
                or not -(2**63) <= line.source_index < 2**63
                or not (
                    type(line.effective_height) is float
                    and math.isfinite(line.effective_height)
                    or type(line.effective_height) is int
                    and -(2**53) <= line.effective_height <= 2**53
                )
            ):
                return None
            records = _plain_source_records(line, geometry, anchors_only=False)
            if records is None:
                return None

            def metadata(positions):
                """只为坐标合法的字符编码字体和 Unicode 类别，与原样本准入顺序一致。"""
                if all(
                    actual is expected
                    for actual, expected in zip(
                        (_font_run_metadata, _script_group, _cached_run_key, _is_anchor_text),
                        _FONT_BATCH_REFERENCE_FUNCTIONS,
                    )
                ):
                    return native.owned_font_run_metadata(
                        line.chars,
                        positions,
                        line.angle,
                        run_ids,
                        run_keys,
                        _font_run_metadata,
                        _script_group,
                        _cached_run_key,
                        _is_anchor_text,
                        layout,
                    )
                cache = _ReadOnlyFontCache(native_only=True)
                output = []
                for position in positions:
                    char = line.chars[position]
                    text = char["char"]
                    font = char.get("font")
                    if font is not None and type(font) is not dict:
                        return None
                    try:
                        key, size = _font_run_key(char, line.angle, text, cache)
                    except _NativeFontUnsupported:
                        return None
                    run_id = run_ids.get(key)
                    if run_id is None:
                        run_id = len(run_keys)
                        run_ids[key] = run_id
                        run_keys.append(key)
                    if layout:
                        output.append((run_id, size, _is_anchor_text(text), char["char_idx"], text))
                    else:
                        output.append((run_id, size, _is_anchor_text(text)))
                return output

            if not document.add_line(
                records,
                page_size,
                line.angle,
                page_index,
                line.source_index,
                line.effective_height,
                metadata,
                line if layout else None,
            ):
                return None
    family_ids = {}
    families = [family_ids.setdefault(key[0], len(family_ids)) for key in run_keys]
    if layout:

        def release_side_maps():
            """原生阶段接受输入后，在最终字符物化之前释放已复制的 loose 侧表。"""
            for geometry in geometries:
                geometry.loose_bboxes.clear()

        result = document.finish_layout(
            families, run_keys, _CharSample, _RunStats, release_side_maps, with_metrics, sys.version_info >= (3, 12)
        )
        if result is None:
            return None
        samples, by_line, runs, inflated, scales = result[:5]
        # 保留旧 set comprehension 的 run 顺序，避免同族 donor 并列时改变选择。
        runs = {key: runs[key] for key in {key for key in run_keys} if key in runs}
        if with_metrics:
            metrics, reports = result[5]
            return samples, by_line, runs, inflated, scales, (metrics, _native_run_diagnostics(run_keys, reports))
        return samples, by_line, runs, inflated, scales
    result = document.finish(families)
    if result is None:
        return None
    inflated, scales, rows = result
    plan = DocumentGeometryPlan(
        style_inflated_runs={run_keys[index] for index in inflated},
        document_style_anomaly=bool(inflated),
        line_style_scales=dict(scales),
    )
    plan.run_diagnostics = _native_run_diagnostics(run_keys, rows)
    # 只有整本文档成功消费后才清理侧表，拒绝输入时仍能完整执行参考路径。
    for geometry in geometries:
        geometry.loose_bboxes.clear()
    return plan


def _native_run_diagnostics(run_keys, rows):
    """仅在输出边界将原生 run 数值转换为既有诊断字段与排序。"""
    diagnostics = []
    for row in sorted(rows, key=lambda row: run_keys[row[0]]):
        index, count, pairs, median_ratio, p90, large_share, overlap_share, strong, sibling, style = row
        diagnostics.append(
            {
                "run_key": list(run_keys[index]),
                "sample_count": count,
                "reliable_pair_count": pairs,
                "median_ratio": median_ratio,
                "p90_ratio": p90,
                "ratio_gt_1_35_share": large_share,
                "next_tight_overlap_share": overlap_share,
                "strong_x_bad": strong,
                "sibling_x_bad": sibling,
                "style_y_bad": style,
            }
        )
    return diagnostics


def _prepare_style_only_document(lines_by_page, geometries, page_sizes):
    """样式分支只生成最终计划，不构造逐字符 Python 对象。"""
    return _prepare_owned_document(lines_by_page, geometries, page_sizes)


def _prepare_layout_document(lines_by_page, geometries, page_sizes, *, with_metrics=False):
    """布局分支只在 X/Y 规则边界物化一次已计算完统计的字符数据。"""
    return _prepare_owned_document(lines_by_page, geometries, page_sizes, layout=True, with_metrics=with_metrics)


def _repair_extreme_local_font_metrics(plan, by_line, page_sizes):
    """用同页重复异常与健康同族行校准极端字体矩阵，不放宽普通跨页样式阈值。"""
    groups = defaultdict(list)
    families = {}

    def family_name(name):
        """同一输入状态的字体后缀只解析一次，缓存仅属于当前文档几何修复。"""
        if name not in families:
            families[name] = re.sub(r"(?:semibold|demibold|regular|regu|bold|light|medium|italic|ital|oblique)+$", "", name)
        return families[name]

    for key, samples in by_line.items():
        anchors = [sample for sample in samples if sample.is_anchor]
        if len(anchors) < 4:
            continue
        dominant = Counter(sample.run_key for sample in anchors).most_common(1)[0][0]
        family = family_name(dominant[0])
        # 描述标志、数字及希腊字符可在同一字体矩阵内变换，不应阻止正文行的尺度校准。
        members = [
            sample
            for sample in anchors
            if family_name(sample.run_key[0]) == family
            and sample.run_key[1] == dominant[1]
            and sample.run_key[4] == dominant[4]
        ]
        if len(members) < 0.8 * len(anchors):
            continue
        height = _quantile([s.local_tight_bbox[3] - s.local_tight_bbox[1] for s in members], 0.75)
        baseline = statistics.median(s.local_origin[1] for s in members)
        if height <= 0 or sum(abs(s.local_origin[1] - baseline) <= 0.2 * height for s in members) < 0.85 * len(members):
            continue
        # 通用强调后缀只用于寻找健康同族几何，不把某个字体名称写成例外。
        groups[key[0], family, dominant[1], dominant[4]].append((key, members, height, baseline))
    for group in groups.values():
        bad = [
            row
            for row in group
            if row[1][0].line.effective_height >= 4 * row[2] and statistics.median(s.font_size for s in row[1]) <= 0.3 * row[2]
        ]
        if len(bad) < 6 or len(bad) < 0.6 * len(group):
            continue
        donors = [row for row in group if len(row[1]) >= 12 and 1.2 <= row[1][0].line.effective_height / row[2] <= 2.5]
        if not donors:
            continue
        ascent = statistics.median((row[3] - row[1][0].local_source_bbox[1]) / row[2] for row in donors)
        descent = statistics.median((row[1][0].local_source_bbox[3] - row[3]) / row[2] for row in donors)
        if ascent <= 0 or descent < 0 or not 1.2 <= ascent + descent <= 2.5:
            continue
        for key, members, height, baseline in bad:
            samples = by_line[key]
            line = members[0].line
            size = page_sizes[key[0]]
            ink = _bbox_union_many([s.tight_bbox for s in samples])
            local_ink = _rotate_bbox_to_upright(ink, size, line.angle)
            local_source = _rotate_bbox_to_upright(line.bbox, size, line.angle)
            local_layout = (
                local_source[0],
                min(local_ink[1], baseline - ascent * height),
                local_source[2],
                max(local_ink[3], baseline + descent * height),
            )
            layout = _clip_bbox(_rotate_bbox_from_upright(local_layout, size, line.angle), size)
            if layout is None:
                continue
            scale = height * (ascent + descent)
            old = plan.line_repairs.get(key)
            plan.line_repairs[key] = LineGeometryRepair(
                source_bbox=line.source_bbox or line.bbox,
                layout_bbox=layout,
                ink_bbox=ink,
                baseline=baseline,
                em_height=scale,
                state="repair_xy" if old and old.state in {"repair_x", "repair_xy"} else "trim_y",
                confidence=1.0,
                run_key=members[0].run_key,
            )
            plan.line_style_scales[key] = scale
            plan.local_metric_repair_sources.add(key)
            # 字符与行共同回到同一基线包络，避免后续脚本/视觉行重分割又读回巨大原生字框。
            for sample in samples:
                if sample.local_source_bbox[3] - sample.local_source_bbox[1] < 4 * height:
                    continue
                old_char = plan.char_repairs.get((key[0], sample.char_idx))
                char_local = _rotate_bbox_to_upright(old_char.layout_bbox if old_char else sample.source_bbox, size, line.angle)
                delta = sample.local_origin[1] - baseline
                char_layout = _clip_bbox(
                    _rotate_bbox_from_upright(
                        (
                            char_local[0],
                            min(local_layout[1] + delta, sample.local_tight_bbox[1]),
                            char_local[2],
                            max(local_layout[3] + delta, sample.local_tight_bbox[3]),
                        ),
                        size,
                        line.angle,
                    ),
                    size,
                )
                if char_layout is None:
                    continue
                plan.char_repairs[key[0], sample.char_idx] = CharLayoutGeometry(
                    source_bbox=sample.source_bbox,
                    tight_bbox=sample.tight_bbox,
                    origin=sample.origin,
                    layout_bbox=char_layout,
                    ink_bbox=sample.tight_bbox,
                    baseline=sample.local_origin[1],
                    advance=old_char.advance if old_char else None,
                    em_height=scale,
                    x_state=old_char.x_state if old_char else "healthy",
                    y_state="abnormal",
                    confidence=1.0,
                )


def build_document_geometry_plan(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
    *,
    owned_geometry_inputs: list[Any] | None = None,
) -> DocumentGeometryPlan:
    """构建文档级 X 修复、Y trim 与 split shadow 计划。"""

    plan = DocumentGeometryPlan()
    if not any(geometry.tight_bboxes and geometry.origins for geometry in geometries):
        return plan
    if owned_geometry_inputs is None:
        # 保持既有单页/测试替身签名；只有主链路显式提供页面级 evidence 才传私有参数。
        risk = _document_requires_full_geometry(lines_by_page, geometries, page_sizes)
    else:
        risk = _document_requires_full_geometry(
            lines_by_page,
            geometries,
            page_sizes,
            owned_geometry_inputs=owned_geometry_inputs,
        )
    if not risk.any:
        for geometry in geometries:
            geometry.loose_bboxes.clear()
        return plan
    if not risk.layout:
        native_plan = _prepare_style_only_document(lines_by_page, geometries, page_sizes)
        if native_plan is not None:
            return native_plan
    prepared = _prepare_layout_document(lines_by_page, geometries, page_sizes, with_metrics=True) if risk.layout else None
    native_y_risk = None
    if prepared is not None:
        samples, by_line, runs, inflated, scales, (metrics, diagnostics) = prepared
        plan.style_inflated_runs = inflated
        plan.document_style_anomaly = bool(inflated)
        plan.line_style_scales.update(scales)
        inks, baselines, native_y_risk = metrics
        plan.line_ink_bboxes.update((key, tuple(box)) for key, box in inks)
        plan.line_baselines.update(baselines)
        plan.run_diagnostics = diagnostics
    else:
        samples, by_line = _collect_samples(lines_by_page, geometries, page_sizes)
        if risk.layout:
            # 只有布局风险才记录公开输出候选；仅样式异常时只校准内部字号。
            _record_line_canonical_metrics(plan, by_line)
        runs, style_inflated_runs, restored = _prepare_run_style(
            plan,
            samples,
            by_line,
            restore_pages=page_sizes if risk.layout else None,
        )
        if plan.document_style_anomaly and risk.layout and not restored:
            _restore_stable_legacy_source_bboxes(by_line, page_sizes)
            runs = _build_run_stats(samples, by_line)
            for run in runs.values():
                run.style_y_bad = run.key in style_inflated_runs
    if prepared is None:
        plan.run_diagnostics = _run_diagnostics(runs)
    if not risk.layout:
        return plan
    x_bad_runs = {run.key for run in runs.values() if run.strong_x_bad or run.sibling_x_bad}
    if x_bad_runs:
        _repair_x_chars(plan, runs, by_line, page_sizes)
        _build_line_repairs_from_x(plan, lines_by_page, by_line, page_sizes)
    if not (native_y_risk if native_y_risk is not None else _has_document_y_risk(by_line)):
        return plan
    analyses = _analyze_lines(lines_by_page, by_line, page_sizes)
    confirmed_runs = _mark_y_candidates(analyses)
    _repair_y_lines(plan, analyses, confirmed_runs, x_bad_runs, page_sizes)
    _repair_extreme_local_font_metrics(plan, by_line, page_sizes)
    return plan


def apply_line_geometry_repairs(
    lines: list[_LineItem],
    *,
    page_index: int,
    plan: DocumentGeometryPlan,
    allow_y_trim: bool,
) -> None:
    """把文档计划应用到当前仍可参与 Flash 布局的行。"""

    for line in lines:
        canonical_ink_bbox = plan.line_ink_bboxes.get(
            (page_index, line.source_index),
        )
        if canonical_ink_bbox is not None:
            line.ink_bbox = canonical_ink_bbox
        canonical_baseline = plan.line_baselines.get(
            (page_index, line.source_index),
        )
        if canonical_baseline is not None:
            line.baseline = canonical_baseline
        style_scale = plan.line_style_scales.get(
            (page_index, line.source_index),
        )
        if style_scale is not None:
            line.em_height = style_scale
        line.style_scale_repaired = (
            page_index,
            line.source_index,
        ) in plan.local_metric_repair_sources or _line_uses_repaired_style_scale(
            line,
            plan.style_inflated_runs,
        )
        repair = plan.line_repairs.get((page_index, line.source_index))
        if repair is None:
            continue
        # 极端局部异常已有健康同族行与六行重复证据，可在表格/图片成员认领前缩回真实行带。
        local_metric_repair = (page_index, line.source_index) in plan.local_metric_repair_sources
        if repair.state in {"trim_y", "repair_xy"} and not allow_y_trim and not local_metric_repair:
            if repair.state == "trim_y":
                continue
            local_state = "repair_x"
            source = repair.source_bbox
            layout = (repair.layout_bbox[0], source[1], repair.layout_bbox[2], source[3])
        else:
            local_state = repair.state
            layout = repair.layout_bbox
        line.source_bbox = repair.source_bbox
        line.ink_bbox = repair.ink_bbox
        if repair.baseline is not None:
            line.baseline = repair.baseline
        line.geometry_state = local_state
        line.geometry_confidence = repair.confidence
        line.split_y_candidate = repair.split_y_candidate
        line.bbox = layout
        line.em_height = style_scale if style_scale is not None else repair.em_height
        if (allow_y_trim or local_metric_repair) and repair.state in {"trim_y", "repair_xy"}:
            line.effective_height = repair.em_height


_FONT_BATCH_REFERENCE_FUNCTIONS = (_font_run_metadata, _script_group, _cached_run_key, _is_anchor_text)


# 固定默认阈值与规则身份；调用方覆盖配置或辅助规则时保持参考语义。
_RISK_CONSTANTS = {
    name: value
    for name, value in globals().copy().items()
    if name.startswith(("X_", "Y_", "ANCHOR_", "STYLE_")) and isinstance(value, (int, float))
}
_RISK_FUNCTIONS = {
    function.__name__: function
    for function in (
        _quantile,
        _style_line_is_inflated,
        _anchor_pair_statistics,
        _prepared_line_geometry,
        _font_run_key,
    )
}

_STYLE_FUNCTIONS = {
    function.__name__: function
    for function in (
        _build_run_stats,
        _mark_style_inflated_runs,
        _document_uses_global_style_calibration,
        _apply_style_scale_repairs,
        _restore_stable_legacy_source_bboxes,
        _CharSample,
        _RunStats,
        _LineItem,
    )
}

_STYLE_ONLY_FUNCTIONS = {
    function.__name__: function
    for function in (
        _collect_samples,
        _record_line_canonical_metrics,
        _has_document_y_risk,
        _baseline_clusters,
        _bbox_union_many,
        _plain_source_records,
        _run_diagnostics,
        _is_anchor_text,
        _font_run_metadata,
        _cached_run_key,
        _script_group,
        _normalized_font_family,
        _ReadOnlyFontCache,
    )
}


__all__ = [
    "CharLayoutGeometry",
    "DocumentGeometryPlan",
    "LineGeometryRepair",
    "apply_line_geometry_repairs",
    "build_document_geometry_plan",
]
