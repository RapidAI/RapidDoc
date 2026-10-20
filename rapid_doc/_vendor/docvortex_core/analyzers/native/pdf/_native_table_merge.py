"""内部自有候选流适配；文字判定保留 Python，数值展开与动态合并连续执行。"""

import math


class _NativeCoreLineSet:
    """以 Rust 核心成员和少量新增注释行提供只读查询及局部选择集合。"""

    def __init__(self, native, extra=None):
        """不枚举核心成员，只保存当前注释扫描新增的少量行号。"""
        self.native = native
        self.extra = set() if extra is None else extra

    def __len__(self):
        """不导出核心集合即可判断大小和真假值。"""
        return len(self.native) + len(self.extra)

    def __iter__(self):
        """仅不支持的参考适配需要显式枚举时物化成员。"""
        yield from self.native.members()
        yield from self.extra

    def __contains__(self, value):
        """注释通常只查询少量来源行，保持核心集合驻留 Rust。"""
        return value in self.extra or self.native.contains_all([value])

    def copy(self):
        """注释扫描拥有独立新增集合，不修改候选核心。"""
        return _NativeCoreLineSet(self.native, self.extra.copy())

    def issuperset(self, values):
        """一次查询整条注释行的来源成员，避免 Python 集合转换整个核心。"""
        return self.native.contains_all([value for value in values if value not in self.extra])

    def update(self, values):
        """仅记录核心之外新选中的注释成员。"""
        self.extra.update(self.native.difference(list(values)))

    def intersection(self, values):
        """仅遍历较小的注释集合，供已有注释几何缓存复用。"""
        values = list(values)
        result = set(self.native.intersection(values))
        if self.extra:
            result.update(self.extra.intersection(values))
        return result


def _plain_box(box):
    """仅准入普通有限框，极端数值保留参考路径的原计算行为。"""
    return (
        type(box) in (tuple, list)
        and len(box) == 4
        and all(type(v) is float and math.isfinite(v) and abs(v) <= 1e100 for v in box)
    )


def _plain_ids(values):
    """确认来源 ID 可无损传入原生有符号整数。"""
    return all(type(v) is int and -(2**63) <= v < 2**63 for v in values)


def _annotations(values):
    """保留注释类型、成员与行框字典插入顺序，不重新裁决注释文字。"""
    return [(a.kind, a.bbox, list(a.line_indices), list(a.line_bboxes.items())) for a in values]


def merge_owned(candidates):
    """仅消费检测器独占的普通候选；特殊输入在执行任何展开前交回参考实现。"""
    from ...._compute_backend import get_native
    from . import table_rules as rules
    from .models import _Fragment, _TableAnnotation, _TableCandidate, _VisualRow, _SharedLineIndexSet

    if len(candidates) < 2:
        return None
    native = get_native()
    if (
        native is None
        or any(getattr(rules, name) is not original for name, original in rules._NATIVE_MERGE_RULES.items())
        or rules._RuleCandidateDraft.materialize is not rules._NATIVE_DRAFT_MATERIALIZE
    ):
        return None
    contexts = {}
    checked_rules = set()
    for candidate in candidates:
        if type(candidate) not in (rules._RuleCandidateDraft, _TableCandidate):
            return None
        if not (
            type(candidate.score) is float
            and math.isfinite(candidate.score)
            or type(candidate.score) is int
            and -(2**53) <= candidate.score <= 2**53
        ):
            return None
        if type(candidate) is rules._RuleCandidateDraft:
            prepared = candidate.prepared_candidate
            if (
                prepared is None
                or type(prepared[0]) is not rules._RuleBandIndex
                or type(prepared[0].native) is not native.PreparedRuleCandidates
            ):
                return None
            for boundary in candidate.boundaries:
                if id(boundary) not in checked_rules:
                    if type(boundary) is not rules._LocalAxisLine or not _plain_box(boundary.bbox):
                        return None
                    checked_rules.add(id(boundary))
            context = candidate.context
            if id(context) in contexts:
                continue
            if (
                type(context) is not rules._RuleCandidateContext
                or type(context.rows) not in (tuple, list)
                or type(context.page_size) not in (tuple, list)
                or type(context.median_height) not in (int, float)
                or abs(context.median_height) > 1e100
                or type(context.angle) is not int
                or not -(2**31) <= context.angle < 2**31
                or not all(type(v) in (int, float) and abs(v) <= 1e100 and math.isfinite(v) for v in context.page_size)
                or len(context.page_size) != 2
                or not math.isfinite(context.median_height)
            ):
                return None
            rows = []
            for row in context.rows:
                if type(row) is not _VisualRow or type(row.fragments) is not list or not _plain_box(row.bbox):
                    return None
                fragments = []
                for fragment in row.fragments:
                    if (
                        type(fragment) is not _Fragment
                        or not _plain_box(fragment.local_bbox)
                        or not _plain_ids([fragment.line_index])
                    ):
                        return None
                    fragments.append((fragment.line_index, fragment.local_bbox))
                rows.append((row.center_y, fragments))
            contexts[id(context)] = (context, rows)
        elif type(candidate) is _TableCandidate:
            if (
                not _plain_box(candidate.bbox)
                or not _plain_box(candidate.local_bbox)
                or (candidate.core_bbox is not None and not _plain_box(candidate.core_bbox))
                or type(candidate.angle) is not int
                or not -(2**31) <= candidate.angle < 2**31
                or type(candidate.line_indices) not in (set, _SharedLineIndexSet)
                or not _plain_ids(candidate.line_indices)
            ):
                return None
            if type(candidate.annotations) is not list:
                return None
            for a in candidate.annotations:
                if (
                    type(a) is not _TableAnnotation
                    or type(a.kind) is not str
                    or not _plain_box(a.bbox)
                    or type(a.line_indices) is not set
                    or type(a.line_bboxes) is not dict
                    or not _plain_ids(a.line_indices)
                    or not _plain_ids(a.line_bboxes)
                    or not all(_plain_box(b) for b in a.line_bboxes.values())
                ):
                    return None
        else:
            return None
    grids = {}
    for key, (context, rows) in contexts.items():
        grid_components = context.grid_components
        if grid_components is not None and not (
            type(grid_components) is list
            and all(
                type(component) is list
                and all(type(rule) is rules._LocalAxisLine and _plain_box(rule.bbox) for rule in component)
                for component in grid_components
            )
        ):
            grid_components = None
        if context.grids is None:
            context.grids = [
                box
                for box in rules._connected_rule_grid_bboxes(
                    context.axis_lines,
                    context.median_height,
                    components=grid_components,
                )
                if not any(rules._bbox_overlap_in_smaller(box, excluded) >= 0.5 for excluded in context.excluded_bboxes)
            ]
        if not all(_plain_box(box) for box in context.grids):
            return None
        grids[key] = native.NativeTableGrid(context.grids, rows, context.page_size, context.angle, context.median_height)
    merger = native.NativeTableMerger()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if type(candidate) is rules._RuleCandidateDraft:
            c = candidate.context
            if c.annotation_geometry is None:
                c.annotation_geometry = rules._prepare_annotation_geometry(c.rows) or False
            bbox, local, core, members, annotations = rules._expand_rule_table_candidate(
                candidate.boundaries,
                candidate.rows,
                c.rows,
                c.lines,
                c.page_size,
                c.angle,
                c.median_height,
                candidate.caption,
                candidate.has_notes,
                c.marker_cache,
                candidate.note_rows,
                c.body_metrics,
                candidate.core_query,
                c.annotation_geometry or None,
                candidate.prepared_candidate,
                owned=True,
            )
            merger.add(bbox, local, c.angle, candidate.score, core, [], _annotations(annotations), members.native, grids[id(c)])
        else:
            merger.add(
                candidate.bbox,
                candidate.local_bbox,
                candidate.angle,
                candidate.score,
                candidate.core_bbox,
                list(candidate.line_indices),
                _annotations(candidate.annotations),
            )
    return [
        _TableCandidate(
            tuple(bbox),
            tuple(local),
            angle,
            score,
            tuple(core) if core is not None else None,
            set(members),
            [
                _TableAnnotation(kind, tuple(box), set(ids), {key: tuple(value) for key, value in rows})
                for kind, box, ids, rows in annotations
            ],
        )
        for bbox, local, angle, score, core, members, annotations in merger.finish()
    ]
