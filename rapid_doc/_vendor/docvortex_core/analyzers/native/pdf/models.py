"""Flash 原生 PDF 提取使用的内部数据模型。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import re

from ....document.pdf._document import PDFPathInfo
from ....document.pdf.text._contracts import Char
from ....schema import BBox

if TYPE_CHECKING:
    from ....document.pdf.form_structure import _PDFFormInfo
    from .inline.types import PDFTextScriptLine

_MARKER_REGEX_REFERENCES = (re.match, re._compile)


class _SharedLineIndexSet(MutableSet[int]):
    """共享只读基底，仅为单个候选记录增删差集，避免复制大型行号集合。"""

    __slots__ = ("base", "added", "removed")

    def __init__(
        self,
        base: frozenset[int],
        values: Iterable[int] = (),
    ) -> None:
        """以共享基底和候选已有成员初始化集合。"""

        self.base = base
        self.removed: set[int] = set()
        if isinstance(values, _SharedLineIndexSet) and values.base is base:
            # 与共享基底求并集时，基底成员天然全部可见，只需保留额外成员。
            self.added = set(values.added)
        elif isinstance(values, (set, frozenset)):
            self.added = values.difference(base)
        else:
            self.added = {value for value in values if value not in base}

    @classmethod
    def from_exact_values(
        cls,
        base: frozenset[int],
        values: Iterable[int],
    ) -> _SharedLineIndexSet:
        """以共享基底精确表示给定集合，而不是默认并入全部基底成员。"""

        visible = set(values)
        instance = cls.__new__(cls)
        instance.base = base
        instance.added = visible.difference(base)
        instance.removed = set(base.difference(visible))
        return instance

    def __contains__(self, value: object) -> bool:
        """按基底、删除集和新增集判断成员是否可见。"""

        if value in self.added:
            return True
        return value in self.base and value not in self.removed

    def __iter__(self) -> Iterator[int]:
        """迭代当前可见成员，不展开持久化副本。"""

        for value in self.base:
            if value not in self.removed:
                yield value
        yield from self.added

    def __len__(self) -> int:
        """返回当前可见成员数量。"""

        return len(self.base) - len(self.removed) + len(self.added)

    def add(self, value: int) -> None:
        """加入成员；基底已有成员只需撤销删除标记。"""

        if value in self.base:
            self.removed.discard(value)
            return
        self.added.add(value)

    def discard(self, value: int) -> None:
        """删除成员；基底成员以差集标记表示。"""

        if value in self.added:
            self.added.discard(value)
            return
        if value in self.base:
            self.removed.add(value)

    def update(self, values: Iterable[int]) -> None:
        """并入成员；同共享基底的候选直接合并差集。"""

        if isinstance(values, _SharedLineIndexSet) and values.base is self.base:
            self.removed.intersection_update(values.removed)
            for value in values.added:
                self.add(value)
            return
        for value in values:
            self.add(value)

    def difference_update(self, values: Iterable[int]) -> None:
        """移除给定成员，并保持共享基底不变。"""

        if isinstance(values, (set, frozenset)):
            self.added.difference_update(values)
            self.removed.update(self.base.intersection(values))
            return
        for value in values:
            self.discard(value)


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其 source/ink/canonical 几何。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    source_bbox: BBox | None = None
    ink_bbox: BBox | None = None
    baseline: float | None = None
    em_height: float = 0.0
    geometry_state: Literal["healthy", "repair_x", "trim_y", "repair_xy", "uncertain"] = "healthy"
    geometry_confidence: float = 1.0
    split_y_candidate: bool = False
    chars: list[Char] = field(default_factory=list)
    visual_row_id: int | None = None
    run_index: int = 0
    effective_height: float = 0.0
    font_signature: tuple[str, int] | None = None
    font_coverage: float = 0.0
    dominant_font_weight: float | None = None
    median_glyph_width: float | None = None
    # 释放原字符后保留少量汉字排版证据，混排 dominant font 改变仍可核验正文连续。
    native_cjk_style: tuple[str, int, float, float] | None = None
    leading_emphasis_width: float | None = None
    leading_typography_width: float | None = None
    split_from_row: bool = False
    preserve_split_boundary: bool = False
    semantic_type: str | None = None
    # 与正文相连的局部短行属于内容，不能因标签在邻页重复而当成页眉。
    marginal_body_context: bool = False
    restored_inline_cluster: bool = False
    compact_formula_cluster: bool = False
    formula_candidate_only: bool = False
    paragraph_formula_context: bool = False
    structural_title: bool = False
    explicit_section_title: bool = False
    title_suppressed: bool = False
    style_scale_repaired: bool = False
    inline_math_regions: list[BBox] = field(default_factory=list)
    paragraph_group: int | None = None
    paragraph_terminal: bool = False
    native_typographic_scale: float | None = field(compare=False, default=None)
    reference_start: bool | None = None
    title_band_id: int | None = None
    caption_start: bool = False
    footnote_marker_start: bool = False
    note_marker_value: str | None = None
    numbered_heading_start: bool = False
    native_math_words: frozenset[str] = field(default_factory=frozenset)
    wrapped_title_parts: tuple[_LineItem, _LineItem] | None = field(compare=False, repr=False, default=None)
    native_title_label_left: float | None = field(compare=False, default=None)
    _marker_features: tuple | None = field(compare=False, repr=False, default=None)
    # 数字图形特征只绑定当前文字对象；几何、语义和认领状态每次重新判断。
    _graphic_numeric_features: tuple | None = field(compare=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """为旧调用与合成测试补齐可选来源几何字段。"""

        if self.source_bbox is None:
            self.source_bbox = self.bbox
        # 数字 run 的角色在输入边界冻结，辅助空间分类只消费编号证据，不重新读取整行文字。
        cached = self._marker_features
        stable_regex = (re.match, re._compile) == _MARKER_REGEX_REFERENCES
        if (
            type(self.text) is str
            and stable_regex
            and type(cached) is tuple
            and len(cached) == 4
            and cached[0] is self.text
            and cached[1] is re.match
        ):
            self.note_marker_value, self.numbered_heading_start = cached[2:]
        else:
            marker = self.text.strip()
            self.note_marker_value = marker if marker.isdigit() and 1 <= len(marker) <= 3 else None
            # 编号标题证据绑定当前行的文字对象；复制只改几何时复用，文字或正则函数改变后重算。
            self.numbered_heading_start = re.match(r"^\d+(?:\.\d+)*\.?(?:\s|$)\S", marker) is not None
            self._marker_features = (
                (self.text, re.match, self.note_marker_value, self.numbered_heading_start)
                if type(self.text) is str and stable_regex
                else None
            )


@dataclass(slots=True)
class _Fragment:
    """保存表格规则使用的单元文本片段。"""

    text: str
    bbox: BBox
    local_bbox: BBox
    line_index: int
    visual_row_id: int | None = None


@dataclass(slots=True)
class _VisualRow:
    """保存同一局部水平带内的表格片段。"""

    fragments: list[_Fragment]
    center_y: float
    bbox: BBox
    visual_row_id: int | None = None


@dataclass(slots=True)
class _AxisLine:
    """保存 PDF 路径中的横竖线。"""

    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(slots=True)
class _LocalAxisLine:
    """保存转入当前文本方向后的横竖线。"""

    bbox: BBox
    original_bbox: BBox
    orientation: Literal["horizontal", "vertical"]
    width: float


@dataclass(slots=True)
class _TableAnnotation:
    """保存表格候选中已识别注释的类型、紧致边界和原始行身份。"""

    kind: Literal["caption", "footnote"]
    bbox: BBox
    line_indices: set[int] = field(default_factory=set)
    line_bboxes: dict[int, BBox] = field(default_factory=dict)


@dataclass(slots=True)
class _TableCandidate:
    """保存已通过相邻横线边界与文本分布校验的表格候选。"""

    bbox: BBox
    local_bbox: BBox
    angle: int
    score: float
    core_bbox: BBox | None = None
    line_indices: MutableSet[int] = field(default_factory=set)
    annotations: list[_TableAnnotation] = field(default_factory=list)
    inferred_grid: list[_AxisLine] = field(default_factory=list)
    # 新恢复的文字网格按原生字符保留强调样式，已有表格维持原物化契约。
    preserve_inline_font_styles: bool = False
    # 已确认/重建的完整线网格不能再混入纯底色矩形的伪行列边界。
    inferred_grid_authoritative: bool = False


@dataclass(slots=True)
class _GraphicCandidate:
    """保存由紧凑绘图线组件形成的图形文本容器候选。"""

    core_bbox: BBox
    lane_index: int
    label_margin_scale: float = 2.5
    line_indices: set[int] = field(default_factory=set)
    # 刻度证明的栅格图已包含完整轴标签，只允许核心内成员，避免再吸入说明续行或图间小框。
    strict_core_members: bool = False


@dataclass(slots=True)
class _DrawingComponentSummary:
    """缓存单页绘图线分量及两处图形检测共用的几何统计。"""

    lines: list[_AxisLine]
    bbox: BBox
    horizontal_count: int
    vertical_count: int


@dataclass(slots=True)
class _CodeCandidate:
    """保存由填充背景或成对横线与稳定文本节奏确认的代码区域。"""

    bbox: BBox
    angle: int
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _TextLane:
    """保存同一文本方向下的局部栏带与已归属文本行。"""

    left: float
    right: float
    lines: list[tuple[_LineItem, BBox]] = field(default_factory=list)
    is_span: bool = False


@dataclass(slots=True)
class _FormulaAnchor:
    """保存公式右缘锚点及其相对正文密集区的上下位置。"""

    line: _LineItem
    bbox: BBox
    detached_below_body: bool = False
    detached_above_body: bool = False
    repeated_number_band: bool = False


@dataclass(frozen=True, slots=True)
class _TableContinuationGrid:
    """仅保留前一物理页末表的方向、归一化列轨和字号容差，不延长字符生命周期。"""

    page_index: int
    angle: int
    positions: tuple[float, ...]
    tolerance: float


@dataclass(slots=True)
class _PageSource:
    """保存单页原生文本分析所需的文本、字符、绘图线和视觉容器。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]
    image_bboxes: list[BBox] = field(default_factory=list)
    signature_bboxes: list[BBox] = field(default_factory=list)
    form_bboxes: list[BBox] = field(default_factory=list)
    path_infos: list[PDFPathInfo] = field(default_factory=list)
    page_index: int | None = None
    publication_bboxes: list[BBox] = field(default_factory=list)
    drawing_component_cache: list[tuple[list[_AxisLine], float, list[_DrawingComponentSummary]]] = field(default_factory=list)
    form_infos: tuple[_PDFFormInfo, ...] = ()
    form_member_sources: dict[BBox, frozenset[int]] = field(default_factory=dict)
    form_path_sources: dict[BBox, frozenset[int]] = field(default_factory=dict)
    retained_page_forms: set[BBox] = field(default_factory=set)
    compound_baseline_cache: dict = field(default_factory=dict)
    isolated_path_cache: dict = field(default_factory=dict)
    previous_table_grid: _TableContinuationGrid | None = None


@dataclass(slots=True)
class _PreparedPage:
    """保存容器认领完成后、等待跨页与文本类型判定的轻量页面。"""

    page_size: tuple[float, float]
    remaining_lines: list[_LineItem]
    table_bboxes: list[BBox]
    drawing_lines: list[_AxisLine]
    fixed_blocks: list[dict[str, Any]]
    # 只保留小圆点候选的轻量路径证据，供完成文本聚合后恢复局部列表。
    bullet_paths: tuple[PDFPathInfo, ...] = ()
    canonical_formula_geometry: bool = False
    canonical_formula_source_lines: list[_LineItem] = field(default_factory=list)
    page_footnote_groups: list[set[int]] = field(default_factory=list)
    reference_regions: list[BBox] = field(default_factory=list)
    numbered_references: bool = False
    # 方括号文献区中的年代/卷页续行不能被小数点编号误作新条目。
    bracketed_reference_numbers: bool = False
    script_lines: list[PDFTextScriptLine] = field(default_factory=list)
    formula_candidate_lines: list[_LineItem] = field(default_factory=list)
    formula_ink_bboxes: list[BBox] = field(default_factory=list)
    # 稀疏表格页保留无字符载荷的原生行样式，防止表体认领后只用大题名反向统计正文。
    table_body_profile_lines: list[_LineItem] = field(default_factory=list)
    local_axis_table_cache: dict[tuple[int, int, int], tuple[list[_LocalAxisLine], list[_LocalAxisLine]]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class _MarginalCandidate:
    """保存页眉页脚带中的单行候选及其正向归一化几何。"""

    page_index: int
    line: _LineItem
    local_bbox: BBox
    local_page_size: tuple[float, float]
    region: Literal["header", "footer", "side"]


@dataclass(slots=True)
class _LaneBodyProfile:
    """保存标题判定使用的栏带正文排版基线，不包含文本内容特征。"""

    body_height: float
    body_font: tuple[str, int] | None
    body_weight: float | None
    regular_gap: float
    style_support: dict[tuple[str, int], float]
    body_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _DocumentBodyProfile:
    """保存跨页正文行高、常规字重与反复出现的常规字体。"""

    body_height: float
    body_weight: float | None
    regular_fonts: frozenset[tuple[str, int]]
    has_style_scale_repairs: bool = False


@dataclass(frozen=True, slots=True)
class _TitleStylePrototype:
    """保存跨页标题原型的字体、尺度、字重和栏内对齐特征。"""

    font_family: str
    font_flags: int
    height_ratio: float
    weight: float | None
    alignment: Literal["left", "center"]
    anchor_offset: float
    support_count: int
    support_pages: int


@dataclass(frozen=True, slots=True)
class _DocumentTitleProfile:
    """保存全文中由重复高置信标题形成的排版原型集合。"""

    prototypes: tuple[_TitleStylePrototype, ...]
