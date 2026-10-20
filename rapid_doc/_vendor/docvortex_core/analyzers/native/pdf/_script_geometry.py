"""提供 PDF 字符 loose/tight/origin 驱动的通用上下标几何分类。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from ....document.pdf.text._contracts import Char
from ....schema import BBox
from ...._compute_backend import get_native

SCRIPT_BODY_COMPARABLE_HEIGHT_RATIO = 0.9
SCRIPT_BASELINE_ABSOLUTE_TOLERANCE = 0.35
SCRIPT_BASELINE_LOOSE_HEIGHT_RATIO = 0.04
SCRIPT_ORIGIN_MIN_SHIFT_ABSOLUTE = 0.5
SCRIPT_ORIGIN_MIN_SHIFT_RATIO = 0.12
SCRIPT_TIGHT_HEIGHT_RATIO = 0.9
SCRIPT_STRONG_SHIFT_RATIO = 0.3
SCRIPT_STRONG_MAX_HEIGHT_RATIO = 1.1
SCRIPT_CONSENSUS_TIGHT_HEIGHT_RATIO = 0.88
SCRIPT_CONSENSUS_ORIGIN_SHIFT_RATIO = 0.22
SCRIPT_CONSENSUS_TIGHT_CENTER_SHIFT_RATIO = 0.3
SCRIPT_CONSENSUS_LOOSE_CENTER_SHIFT_RATIO = 0.15
SCRIPT_LOOSE_HEIGHT_ANOMALY_RATIO = 1.35
SCRIPT_COMPONENT_FORWARD_GAP_RATIO = 1.5
SCRIPT_COMPONENT_X_BACKTRACK_RATIO = 0.5
SCRIPT_COMPONENT_SEED_GAP_RATIO = 0.5
SCRIPT_COMPONENT_SEED_MAX_POSITION_DISTANCE = 4
SCRIPT_COMPONENT_MIN_OFFSET_RATIO = 0.08
SCRIPT_COMPONENT_MAX_HEIGHT_RATIO = 1.1
CONTROL_LINE_BREAK_CHARS = {"\r", "\n"}
_NUMERIC_SCRIPT_SEPARATORS = frozenset({",", "，", "-", "–", "—"})

ScriptRole = Literal["body", "sup", "sub"]
ScriptMarkRole = Literal["sup", "sub"]
_ScriptFontKey = tuple[str, int | None, int | None]


@dataclass(frozen=True, slots=True)
class ScriptCharFeature:
    """保存单个字符参与纯几何上下标判定所需的只读特征。"""

    index: int
    text: str
    loose_bbox: BBox
    tight_bbox: BBox | None
    origin: tuple[float, float] | None
    is_valid: bool
    is_body_anchor: bool

    @property
    def loose_height(self) -> float:
        """返回 loose bbox 高度。"""
        return self.loose_bbox[3] - self.loose_bbox[1]

    @property
    def loose_center_y(self) -> float:
        """返回 loose bbox 中心 y。"""
        return (self.loose_bbox[1] + self.loose_bbox[3]) / 2

    @property
    def tight_height(self) -> float:
        """返回 tight bbox 高度，无有效框时返回零。"""
        if self.tight_bbox is None:
            return 0.0
        return self.tight_bbox[3] - self.tight_bbox[1]

    @property
    def tight_center_y(self) -> float | None:
        """返回 tight bbox 中心 y。"""
        if self.tight_bbox is None:
            return None
        return (self.tight_bbox[1] + self.tight_bbox[3]) / 2


@dataclass(frozen=True, slots=True)
class ScriptBodyBand:
    """表示当前视觉组件的正文 origin 基线与双 bbox 参考高度。"""

    baseline: float
    tight_height: float
    loose_height: float
    member_indices: frozenset[int]


@dataclass(frozen=True, slots=True)
class ScriptBaselineCluster:
    """表示共享近似字符 origin 的正文或角标基线簇。"""

    baseline: float
    member_indices: tuple[int, ...]


def _coerce_finite_bbox(value: Any) -> BBox | None:
    """把可迭代四元组收敛为合法有限 bbox。"""
    try:
        raw_bbox = getattr(value, "bbox", value)
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _char_geometry_key(char: Char) -> int | None:
    """返回可用于 side-map 查询的合法 PDFium char_idx。"""
    char_idx = char.get("char_idx")
    if isinstance(char_idx, bool) or not isinstance(char_idx, int):
        return None
    return char_idx


def _script_role(offset: float) -> ScriptMarkRole:
    """把相对正文 origin 基线的纵向偏移转换为上下标角色。"""
    return "sub" if offset > 0 else "sup"


def build_script_features(
    chars: list[Char],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    protected_body_indices: set[int],
) -> list[ScriptCharFeature]:
    """一次性构造 loose/tight/origin 三类字符几何特征。"""
    features = []
    for index, char in enumerate(chars):
        text = str(char.get("char", ""))
        char_idx = _char_geometry_key(char)
        loose_bbox = _coerce_finite_bbox(char.get("bbox")) or (0.0, 0.0, 0.0, 0.0)
        tight_bbox = _coerce_finite_bbox(tight_bboxes.get(char_idx)) if char_idx is not None else None
        raw_origin = origins.get(char_idx) if char_idx is not None else None
        origin = None
        if raw_origin is not None:
            try:
                candidate_origin = (float(raw_origin[0]), float(raw_origin[1]))
            except (IndexError, TypeError, ValueError):
                candidate_origin = None
            if candidate_origin is not None and all(math.isfinite(value) for value in candidate_origin):
                origin = candidate_origin
        is_valid = text not in CONTROL_LINE_BREAK_CHARS and not text.isspace() and tight_bbox is not None and origin is not None
        features.append(
            ScriptCharFeature(
                index=index,
                text=text,
                loose_bbox=loose_bbox,
                tight_bbox=tight_bbox,
                origin=origin,
                is_valid=is_valid,
                is_body_anchor=is_valid and text.isalnum() and index not in protected_body_indices,
            )
        )
    return features


def split_script_visual_components(features: list[ScriptCharFeature]) -> list[list[int]]:
    """按换行、x 回退和大间隙切分独立视觉组件。"""
    valid_heights = [feature.loose_height for feature in features if feature.loose_height > 0]
    scale = statistics.median(valid_heights) if valid_heights else 1.0
    components: list[list[int]] = []
    current: list[int] = []
    previous_visible: ScriptCharFeature | None = None
    for feature in features:
        if feature.text in CONTROL_LINE_BREAK_CHARS or feature.text.isspace():
            if current:
                components.append(current)
                current = []
            previous_visible = None
            continue
        if feature.is_valid and previous_visible is not None:
            x_backtrack = previous_visible.loose_bbox[0] - feature.loose_bbox[0]
            forward_gap = feature.loose_bbox[0] - previous_visible.loose_bbox[2]
            if (
                x_backtrack > scale * SCRIPT_COMPONENT_X_BACKTRACK_RATIO
                or forward_gap > scale * SCRIPT_COMPONENT_FORWARD_GAP_RATIO
            ):
                if current:
                    components.append(current)
                current = []
                previous_visible = None
        current.append(feature.index)
        if feature.is_valid:
            previous_visible = feature
    if current:
        components.append(current)
    return components


def _quantile(values: list[float], fraction: float) -> float:
    """返回适合小样本基线簇的稳定分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _cluster_baselines(
    features: list[ScriptCharFeature],
    component_indices: list[int],
) -> tuple[list[ScriptBaselineCluster], float]:
    """按字符 origin y 聚类当前组件的字母数字基线。"""
    anchors = [features[index] for index in component_indices if features[index].is_body_anchor]
    if not anchors:
        return [], 0.0
    median_loose_height = statistics.median(feature.loose_height for feature in anchors)
    tolerance = max(SCRIPT_BASELINE_ABSOLUTE_TOLERANCE, median_loose_height * SCRIPT_BASELINE_LOOSE_HEIGHT_RATIO)
    groups: list[list[int]] = []
    group_origins: list[list[float]] = []
    for feature in sorted(anchors, key=lambda item: item.origin[1] if item.origin is not None else 0.0):
        baseline = feature.origin[1] if feature.origin is not None else 0.0
        if not groups:
            groups.append([feature.index])
            group_origins.append([baseline])
            continue
        previous_baseline = _sorted_origin_median(group_origins[-1])
        if abs(baseline - previous_baseline) <= tolerance:
            groups[-1].append(feature.index)
            group_origins[-1].append(baseline)
        else:
            groups.append([feature.index])
            group_origins.append([baseline])
    return (
        [
            ScriptBaselineCluster(
                baseline=_sorted_origin_median(origins),
                member_indices=tuple(group),
            )
            for group, origins in zip(groups, group_origins, strict=True)
        ],
        tolerance,
    )


def _sorted_origin_median(origins: list[float]) -> float:
    """原始字符已按 origin 排序，直接使用与 statistics.median 相同的中位运算。"""
    middle = len(origins) // 2
    return origins[middle] if len(origins) % 2 else (origins[middle - 1] + origins[middle]) / 2


def _cluster_tight_height(features: list[ScriptCharFeature], cluster: ScriptBaselineCluster) -> float:
    """返回基线簇的高分位 tight 高度。"""
    return _quantile([height for index in cluster.member_indices if (height := features[index].tight_height) > 0], 0.9)


def _cluster_loose_height(features: list[ScriptCharFeature], cluster: ScriptBaselineCluster) -> float:
    """返回基线簇的高分位 loose 高度。"""
    return _quantile([height for index in cluster.member_indices if (height := features[index].loose_height) > 0], 0.9)


def _cluster_tight_center(features: list[ScriptCharFeature], cluster: ScriptBaselineCluster) -> float:
    """返回基线簇的 tight bbox 中心中位数。"""
    return statistics.median(
        features[index].tight_center_y for index in cluster.member_indices if features[index].tight_center_y is not None
    )


def _cluster_loose_center(features: list[ScriptCharFeature], cluster: ScriptBaselineCluster) -> float:
    """返回基线簇的 loose bbox 中心中位数。"""
    return statistics.median(features[index].loose_center_y for index in cluster.member_indices)


def _cluster_has_consistent_displacement(
    features: list[ScriptCharFeature],
    cluster: ScriptBaselineCluster,
    body_cluster: ScriptBaselineCluster,
    body_band: ScriptBodyBand,
) -> bool:
    """要求 origin 与双 bbox 至少两项同向，并排除普通混合字体的弱中心偏移。"""
    origin_shift = cluster.baseline - body_band.baseline
    tight_shift = _cluster_tight_center(features, cluster) - _cluster_tight_center(features, body_cluster)
    loose_shift = _cluster_loose_center(features, cluster) - _cluster_loose_center(features, body_cluster)
    shifts = (origin_shift, tight_shift, loose_shift)
    expected_positive = origin_shift > 0
    if sum(value > 0.05 if expected_positive else value < -0.05 for value in shifts) < 2:
        return False
    origin_ratio = abs(origin_shift) / max(body_band.tight_height, 1e-6)
    tight_ratio = abs(tight_shift) / max(body_band.tight_height, 1e-6)
    loose_ratio = abs(loose_shift) / max(body_band.loose_height, 1e-6)
    return origin_ratio >= SCRIPT_CONSENSUS_ORIGIN_SHIFT_RATIO or (
        tight_ratio >= SCRIPT_CONSENSUS_TIGHT_CENTER_SHIFT_RATIO and loose_ratio >= SCRIPT_CONSENSUS_LOOSE_CENTER_SHIFT_RATIO
    )


def _choose_body_band(
    features: list[ScriptCharFeature],
    clusters: list[ScriptBaselineCluster],
    heights: dict[ScriptBaselineCluster, tuple[float, float]] | None = None,
) -> tuple[ScriptBodyBand, ScriptBaselineCluster] | None:
    """在接近最高字形的基线簇中按字符数选择正文基线。"""
    if not clusters:
        return None
    if heights is None:
        heights = {
            cluster: (_cluster_tight_height(features, cluster), _cluster_loose_height(features, cluster))
            for cluster in clusters
        }
    maximum_height = max(heights[cluster][0] for cluster in clusters)
    comparable = [
        cluster for cluster in clusters if heights[cluster][0] >= maximum_height * SCRIPT_BODY_COMPARABLE_HEIGHT_RATIO
    ]
    body_cluster = max(
        comparable,
        key=lambda cluster: (
            len(cluster.member_indices),
            heights[cluster][0],
            heights[cluster][1],
        ),
    )
    return (
        ScriptBodyBand(
            baseline=body_cluster.baseline,
            tight_height=heights[body_cluster][0],
            loose_height=heights[body_cluster][1],
            member_indices=frozenset(body_cluster.member_indices),
        ),
        body_cluster,
    )


def _nearest_cluster(
    feature: ScriptCharFeature,
    clusters: list[ScriptBaselineCluster],
    tolerance: float,
) -> ScriptBaselineCluster | None:
    """把非字母数字字符附着到最近 origin 基线簇。"""
    if feature.origin is None or not clusters:
        return None
    nearest = min(clusters, key=lambda cluster: abs(feature.origin[1] - cluster.baseline))
    return nearest if abs(feature.origin[1] - nearest.baseline) <= tolerance else None


def _horizontal_gap(first: BBox, second: BBox) -> float:
    """返回两个字符 tight bbox 的水平间隙。"""
    return max(0.0, first[0] - second[2], second[0] - first[2])


def _drop_unseeded_punctuation(
    features: list[ScriptCharFeature],
    component_indices: list[int],
    roles: list[ScriptRole],
    body_height: float,
) -> None:
    """移除未邻近同类字母数字种子的标点。"""
    seeded = [
        features[index]
        for index in component_indices
        if roles[index] != "body" and features[index].text.isalnum() and features[index].tight_bbox is not None
    ]
    for index in component_indices:
        feature = features[index]
        role = roles[index]
        if role == "body" or feature.text.isalnum() or feature.tight_bbox is None:
            continue
        nearby = any(
            roles[seed.index] == role
            and _horizontal_gap(feature.tight_bbox, seed.tight_bbox) <= max(2.0, body_height * SCRIPT_COMPONENT_SEED_GAP_RATIO)
            and abs(feature.index - seed.index) <= SCRIPT_COMPONENT_SEED_MAX_POSITION_DISTANCE
            for seed in seeded
        )
        if not nearby:
            roles[index] = "body"


def _apply_consensus_candidates(
    features: list[ScriptCharFeature],
    component_indices: list[int],
    body_band: ScriptBodyBand,
    roles: list[ScriptRole],
) -> None:
    """用 origin/tight/loose 三证据一致性补充孤立边界字符。"""
    body_members = [
        features[index]
        for index in body_band.member_indices
        if features[index].tight_bbox is not None
        and features[index].tight_center_y is not None
        and features[index].origin is not None
    ]
    for index in component_indices:
        feature = features[index]
        if (
            roles[index] != "body"
            or index in body_band.member_indices
            or not feature.is_valid
            or feature.tight_bbox is None
            or feature.tight_center_y is None
            or feature.origin is None
            or not body_members
        ):
            continue
        reference = min(
            body_members,
            key=lambda member: (_horizontal_gap(feature.tight_bbox, member.tight_bbox), abs(feature.index - member.index)),
        )
        if _horizontal_gap(feature.tight_bbox, reference.tight_bbox) > max(2.5, reference.tight_height * 1.2):
            continue
        if feature.tight_height / max(reference.tight_height, 1e-6) > SCRIPT_CONSENSUS_TIGHT_HEIGHT_RATIO:
            continue
        shifts = (
            feature.origin[1] - reference.origin[1],
            feature.tight_center_y - reference.tight_center_y,
            feature.loose_center_y - reference.loose_center_y,
        )
        positive_votes = sum(value > 0.05 for value in shifts)
        negative_votes = sum(value < -0.05 for value in shifts)
        if max(positive_votes, negative_votes) < 2:
            continue
        role = "sub" if positive_votes > negative_votes else "sup"
        if _script_role(shifts[0]) != role:
            continue
        origin_ratio = abs(shifts[0]) / max(reference.tight_height, 1e-6)
        tight_ratio = abs(shifts[1]) / max(reference.tight_height, 1e-6)
        loose_ratio = abs(shifts[2]) / max(reference.loose_height, 1e-6)
        if not feature.text.isalnum():
            expected_positive = role == "sub"
            if not all(value > 0.05 if expected_positive else value < -0.05 for value in shifts) or not (
                origin_ratio >= SCRIPT_STRONG_SHIFT_RATIO
                and tight_ratio >= SCRIPT_CONSENSUS_TIGHT_CENTER_SHIFT_RATIO
                and loose_ratio >= SCRIPT_CONSENSUS_LOOSE_CENTER_SHIFT_RATIO
            ):
                continue
        if origin_ratio >= SCRIPT_CONSENSUS_ORIGIN_SHIFT_RATIO or (
            tight_ratio >= SCRIPT_CONSENSUS_TIGHT_CENTER_SHIFT_RATIO
            and loose_ratio >= SCRIPT_CONSENSUS_LOOSE_CENTER_SHIFT_RATIO
        ):
            roles[index] = role


def _expand_component_neighbors(
    features: list[ScriptCharFeature],
    component_indices: list[int],
    body_band: ScriptBodyBand,
    protected_body_indices: set[int],
    roles: list[ScriptRole],
) -> None:
    """把同侧连续字符并入已有角标 run。"""
    component_set = set(component_indices)
    changed = True
    while changed:
        changed = False
        for index in component_indices:
            feature = features[index]
            if (
                roles[index] != "body"
                or index in protected_body_indices
                or not feature.is_valid
                or feature.origin is None
                or feature.text.isspace()
                or feature.tight_height > body_band.tight_height * SCRIPT_COMPONENT_MAX_HEIGHT_RATIO
            ):
                continue
            neighbor_roles = {
                roles[neighbor]
                for neighbor in (index - 1, index + 1)
                if neighbor in component_set and roles[neighbor] != "body"
            }
            if len(neighbor_roles) != 1:
                continue
            role = next(iter(neighbor_roles))
            shift = feature.origin[1] - body_band.baseline
            if abs(shift) < body_band.tight_height * SCRIPT_COMPONENT_MIN_OFFSET_RATIO or _script_role(shift) != role:
                continue
            roles[index] = role
            changed = True


def _script_font_key(char: Char) -> _ScriptFontKey | None:
    """读取原始字体身份，不合并子集名称，也不把原始字号当作有效字形高度。"""
    font = char.get("font")
    if not isinstance(font, dict):
        return None
    name = font.get("name")
    if not isinstance(name, str) or not name:
        return None
    flags, weight = font.get("flags"), font.get("weight")
    return name, flags if type(flags) is int else None, weight if type(weight) is int else None


def _local_font_run(
    features: list[ScriptCharFeature],
    font_keys: list[_ScriptFontKey | None],
    component: set[int],
    cluster: ScriptBaselineCluster,
) -> set[int]:
    """限定同一视觉组件内连续的同字体 ASCII/全角英数字串。"""
    keys = {font_keys[index] for index in cluster.member_indices}
    if len(keys) != 1 or None in keys:
        return set()
    key = next(iter(keys))

    def belongs(index: int) -> bool:
        """让空白、括号、运算符、CJK、缺失几何和字体边界阻断局部参考。"""
        if index not in component or not features[index].is_valid or font_keys[index] != key:
            return False
        text = features[index].text
        return len(text) == 1 and (
            (text.isascii() and text.isalnum()) or "０" <= text <= "９" or "Ａ" <= text <= "Ｚ" or "ａ" <= text <= "ｚ"
        )

    left = min(cluster.member_indices)
    right = left + 1
    while belongs(left - 1):
        left -= 1
    while belongs(right):
        right += 1
    if not all(left <= index < right and belongs(index) for index in cluster.member_indices):
        return set()
    return set(range(left, right))


def _recheck_weak_clusters_with_local_font_body(
    features: list[ScriptCharFeature],
    font_keys: list[_ScriptFontKey | None],
    component_indices: list[int],
    body_band: ScriptBodyBand,
    cluster_roles: dict[ScriptBaselineCluster, ScriptRole],
    heights: dict[ScriptBaselineCluster, tuple[float, float]],
) -> None:
    """用同字体局部正文撤销弱误判；不制造角标，不让撤销结果成为新参考。"""
    initial_roles = dict(cluster_roles)
    component = set(component_indices)
    for cluster, role in initial_roles.items():
        if role == "body" or abs(cluster.baseline - body_band.baseline) >= body_band.tight_height * SCRIPT_STRONG_SHIFT_RATIO:
            continue
        run = _local_font_run(features, font_keys, component, cluster)
        if not run:
            continue
        references: list[tuple[float, int, ScriptBaselineCluster]] = []
        for reference, reference_role in initial_roles.items():
            if reference_role != "body":
                continue
            indices = tuple(index for index in reference.member_indices if index in run)
            if len(indices) < 2:
                continue
            gap = min(
                _horizontal_gap(features[first].tight_bbox, features[second].tight_bbox)
                for first in cluster.member_indices
                for second in indices
                if features[first].tight_bbox is not None and features[second].tight_bbox is not None
            )
            if gap > max(2.5, body_band.tight_height * 1.2):
                continue
            baseline = statistics.median(features[index].origin[1] for index in indices if features[index].origin is not None)
            references.append((gap, -len(indices), ScriptBaselineCluster(baseline, indices)))
        if not references:
            continue
        reference = min(references, key=lambda item: item[:2])[2]
        height = _cluster_tight_height(features, reference)
        if height <= 0:
            continue
        shift = cluster.baseline - reference.baseline
        strong_shift = abs(shift) >= height * SCRIPT_STRONG_SHIFT_RATIO
        tight_ratio = heights[cluster][0] / height
        if abs(shift) < max(SCRIPT_ORIGIN_MIN_SHIFT_ABSOLUTE, height * SCRIPT_ORIGIN_MIN_SHIFT_RATIO) or (
            tight_ratio > SCRIPT_TIGHT_HEIGHT_RATIO and not (strong_shift and tight_ratio <= SCRIPT_STRONG_MAX_HEIGHT_RATIO)
        ):
            cluster_roles[cluster] = "body"


def _assign_component(
    features: list[ScriptCharFeature],
    component_indices: list[int],
    protected_body_indices: set[int],
    roles: list[ScriptRole],
    font_keys: list[_ScriptFontKey | None],
) -> None:
    """在单个视觉组件内按 origin 基线簇和双 bbox 一致性分配角色。"""
    clusters, tolerance = _cluster_baselines(features, component_indices)
    # 基线簇在本次组件判定内不变，高度分位数可供正文选择与各角标规则共享。
    heights = {
        cluster: (_cluster_tight_height(features, cluster), _cluster_loose_height(features, cluster)) for cluster in clusters
    }
    body_result = _choose_body_band(features, clusters, heights)
    if body_result is None:
        return
    body_band, body_cluster = body_result
    if body_band.tight_height <= 0 or body_band.loose_height <= 0:
        return
    cluster_roles: dict[ScriptBaselineCluster, ScriptRole] = {body_cluster: "body"}
    for cluster in clusters:
        if cluster is body_cluster:
            continue
        shift = cluster.baseline - body_band.baseline
        tight_ratio = heights[cluster][0] / body_band.tight_height
        loose_ratio = heights[cluster][1] / body_band.loose_height
        minimum_shift = max(SCRIPT_ORIGIN_MIN_SHIFT_ABSOLUTE, body_band.tight_height * SCRIPT_ORIGIN_MIN_SHIFT_RATIO)
        strong_shift = abs(shift) >= body_band.tight_height * SCRIPT_STRONG_SHIFT_RATIO
        if (
            abs(shift) < minimum_shift
            or (
                tight_ratio > SCRIPT_TIGHT_HEIGHT_RATIO and not (strong_shift and tight_ratio <= SCRIPT_STRONG_MAX_HEIGHT_RATIO)
            )
            or (
                loose_ratio > SCRIPT_LOOSE_HEIGHT_ANOMALY_RATIO
                and not _cluster_has_consistent_displacement(features, cluster, body_cluster, body_band)
            )
        ):
            cluster_roles[cluster] = "body"
        else:
            cluster_roles[cluster] = _script_role(shift)
    _recheck_weak_clusters_with_local_font_body(features, font_keys, component_indices, body_band, cluster_roles, heights)
    for index in component_indices:
        feature = features[index]
        if (
            index in protected_body_indices
            or feature.origin is None
            or feature.text.isspace()
            or feature.text in CONTROL_LINE_BREAK_CHARS
        ):
            continue
        cluster = _nearest_cluster(feature, clusters, tolerance)
        if cluster is not None:
            roles[index] = cluster_roles.get(cluster, "body")
    _drop_unseeded_punctuation(features, component_indices, roles, body_band.tight_height)
    _apply_consensus_candidates(features, component_indices, body_band, roles)
    _expand_component_neighbors(features, component_indices, body_band, protected_body_indices, roles)
    for index in protected_body_indices.intersection(component_indices):
        roles[index] = "body"


def paired_script_roles(
    chars: list[Char],
    roles: list[ScriptRole],
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> dict[int, ScriptRole]:
    """确认同一基字符右侧横向叠放的上下标，排除分式和远距邻列。"""

    features = build_script_features(chars, tight_bboxes, origins, set())
    visible = [feature for feature in features if feature.text.isprintable() and not feature.text.isspace()]
    paired: dict[int, ScriptRole] = {}
    for position, (base, first, second) in enumerate(zip(visible, visible[1:], visible[2:])):
        if (
            not all(feature.is_valid and feature.text.isalnum() for feature in (base, first, second))
            or roles[base.index] != "body"
            or {roles[first.index], roles[second.index]} != {"sup", "sub"}
        ):
            continue
        assert base.tight_bbox is not None and base.origin is not None
        assert first.tight_bbox is not None and first.origin is not None
        assert second.tight_bbox is not None and second.origin is not None
        scale = base.tight_height
        if scale <= 0 or any(feature.tight_height > 0.8 * scale for feature in (first, second)):
            continue
        overlap = min(first.tight_bbox[2], second.tight_bbox[2]) - max(first.tight_bbox[0], second.tight_bbox[0])
        minimum_width = min(first.tight_bbox[2] - first.tight_bbox[0], second.tight_bbox[2] - second.tight_bbox[0])
        if overlap < 0.5 * minimum_width:
            continue
        if any(
            not -0.15 * scale <= feature.tight_bbox[0] - base.tight_bbox[2] <= 0.5 * scale
            or abs(feature.origin[1] - base.origin[1]) < 0.15 * scale
            or abs(feature.origin[1] - base.origin[1]) > scale
            for feature in (first, second)
        ):
            continue
        if (first.origin[1] - base.origin[1]) * (second.origin[1] - base.origin[1]) >= 0:
            continue
        # 本补证据只接纳完整单字符角标，不能只给 grid 等多字母角标的首字母加样式。
        following = visible[position + 3] if position + 3 < len(visible) else None
        if (
            following is not None
            and following.is_valid
            and following.text.isalnum()
            and following.tight_bbox is not None
            and following.origin is not None
            and following.tight_height <= 0.8 * scale
            and abs(following.origin[1] - second.origin[1]) <= 0.15 * scale
            and -0.15 * scale <= following.tight_bbox[0] - second.tight_bbox[2] <= 0.5 * scale
        ):
            continue
        paired[first.index] = roles[first.index]
        paired[second.index] = roles[second.index]
    return paired


def _numeric_superscript_indices(
    features: list[ScriptCharFeature],
    roles: list[ScriptRole],
    protected_body_indices: set[int],
) -> set[int]:
    """用左侧同一视觉行的稳定正文确认完整数字上标，允许中间有排版空白。"""
    accepted: set[int] = set()
    start = 0
    while start < len(features):
        if not features[start].text.isdecimal():
            start += 1
            continue
        end = start + 1
        while end < len(features) and (features[end].text.isdecimal() or features[end].text in _NUMERIC_SCRIPT_SEPARATORS):
            end += 1
        # 尾随标点属于正文，只有数字之间且共享上标基线的分隔符才随引用提升。
        while end > start and not features[end - 1].text.isdecimal():
            end -= 1
        candidate = features[start:end]
        if any(0 <= neighbor < len(features) and features[neighbor].text in {"/", "⁄"} for neighbor in (start - 1, end)):
            # 不能把分式的分子或分母独立当成引用，否则会制造仅半个指数带上标的结果。
            start = end
            continue
        previous = start - 1
        while previous >= 0 and features[previous].text.isspace():
            if features[previous].text in CONTROL_LINE_BREAK_CHARS:
                break
            previous -= 1
        reference_end = previous + 1
        while previous >= 0 and features[previous].is_valid and roles[previous] == "body":
            if previous in protected_body_indices:
                break
            previous -= 1
        reference = list(range(previous + 1, reference_end))
        start = end
        if not reference or any(
            not item.is_valid or item.index in protected_body_indices or roles[item.index] == "sub" for item in candidate
        ):
            continue
        clusters, _ = _cluster_baselines(features, reference)
        body_result = _choose_body_band(features, clusters)
        if body_result is None:
            continue
        band, body_cluster = body_result
        if len(band.member_indices) < 2 or band.tight_height <= 0 or band.loose_height <= 0:
            continue
        edge = features[reference[-1]]
        first = candidate[0]
        assert first.tight_bbox is not None and edge.tight_bbox is not None
        gap = first.tight_bbox[0] - edge.tight_bbox[2]
        if not -0.15 * band.tight_height <= gap <= max(2.5, 1.2 * band.tight_height):
            continue
        digits = [item for item in candidate if item.text.isdecimal()]
        baseline = statistics.median(item.origin[1] for item in digits if item.origin is not None)
        shift = band.baseline - baseline
        if (
            not max(SCRIPT_ORIGIN_MIN_SHIFT_ABSOLUTE, SCRIPT_CONSENSUS_ORIGIN_SHIFT_RATIO * band.tight_height)
            <= shift
            <= (0.8 * band.tight_height)
        ):
            continue
        tolerance = max(SCRIPT_BASELINE_ABSOLUTE_TOLERANCE, SCRIPT_BASELINE_LOOSE_HEIGHT_RATIO * band.loose_height)
        if any(item.origin is None or abs(item.origin[1] - baseline) > tolerance for item in candidate):
            continue
        tight_ratio = max(item.tight_height for item in digits) / band.tight_height
        if tight_ratio > SCRIPT_CONSENSUS_TIGHT_HEIGHT_RATIO:
            # 纯小写词的 x-height 可接近缩小数字；强基线位移还须有 loose 缩小证据，不能仅放宽 tight 阈值。
            loose_ratio = max(item.loose_height for item in digits) / band.loose_height
            if not (
                shift >= SCRIPT_STRONG_SHIFT_RATIO * band.tight_height
                and tight_ratio <= SCRIPT_STRONG_MAX_HEIGHT_RATIO
                and loose_ratio <= SCRIPT_CONSENSUS_TIGHT_HEIGHT_RATIO
            ):
                continue
        digit_cluster = ScriptBaselineCluster(baseline, tuple(item.index for item in digits))
        if (
            _cluster_tight_center(features, digit_cluster) >= _cluster_tight_center(features, body_cluster) - 0.05
            or _cluster_loose_center(features, digit_cluster) >= _cluster_loose_center(features, body_cluster) - 0.05
        ):
            continue
        # 只补样式，不合并组件或移动字符；下一候选也不能把本次提升结果当作正文。
        accepted.update(item.index for item in candidate)
    return accepted


@lru_cache(maxsize=4096)
def _native_text_flags(text: str) -> int:
    """缓存 Unicode 字符类别值，不缓存文档对象或可变几何。"""
    local = len(text) == 1 and (
        (text.isascii() and text.isalnum()) or "０" <= text <= "９" or "Ａ" <= text <= "Ｚ" or "ａ" <= text <= "ｚ"
    )
    return (
        int(text.isspace())
        | (int(text in CONTROL_LINE_BREAK_CHARS) << 1)
        | (int(text.isalnum()) << 2)
        | (int(text.isdecimal()) << 3)
        | (int(local) << 4)
        | (int(text in {"/", "⁄"}) << 5)
        | (int(text in _NUMERIC_SCRIPT_SEPARATORS) << 6)
    )


def _native_script_inputs(chars, tight_bboxes, origins, protected):
    """打包一次调用的几何与 Python 字体等价类，保留异常输入的原校验语义。"""
    loose_boxes, tight_boxes, points, flags_list, font_ids = [], [], [], [], []
    fonts = {}
    for index, char in enumerate(chars):
        text = str(char.get("char", ""))
        char_idx = _char_geometry_key(char)
        loose = char.get("bbox")
        loose_boxes.append(getattr(loose, "bbox", loose))
        tight_boxes.append(tight_bboxes.get(char_idx) if char_idx is not None else None)
        raw_origin = origins.get(char_idx) if char_idx is not None else None
        origin = None
        if raw_origin is not None:
            try:
                candidate = (float(raw_origin[0]), float(raw_origin[1]))
            except (IndexError, TypeError, ValueError):
                candidate = None
            origin = candidate
        flags = _native_text_flags(text)
        flags |= int(index in protected) << 7
        font = _script_font_key(char)
        font_id = -1 if font is None else fonts.setdefault(font, len(fonts))
        points.append(origin)
        flags_list.append(flags)
        font_ids.append(font_id)
    return loose_boxes, tight_boxes, points, flags_list, font_ids


def classify_char_script_roles(
    chars: list[Char],
    *,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    protected_body_indices: set[int] | None = None,
) -> list[ScriptRole]:
    """按视觉组件、origin 基线簇和双 bbox 一致性识别上下标。"""
    native = get_native()
    if native is not None:
        roles = native.script_roles_raw(
            *_native_script_inputs(chars, tight_bboxes, origins, protected_body_indices or set()), _coerce_finite_bbox
        )
        if roles is not None:
            names = ("body", "sup", "sub")
            return [names[role] for role in roles]
    return _classify_char_script_roles_python(
        chars, tight_bboxes=tight_bboxes, origins=origins, protected_body_indices=protected_body_indices
    )


def _classify_char_script_roles_python(
    chars: list[Char],
    *,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    protected_body_indices: set[int] | None = None,
) -> list[ScriptRole]:
    """保留原逐字符算法作为差分参考及集合平局的精确处理路径。"""
    protected = protected_body_indices or set()
    features = build_script_features(chars, tight_bboxes, origins, protected)
    font_keys = [_script_font_key(char) for char in chars]
    roles: list[ScriptRole] = ["body"] * len(features)
    for component_indices in split_script_visual_components(features):
        _assign_component(features, component_indices, protected, roles, font_keys)
    for index in _numeric_superscript_indices(features, roles, protected):
        roles[index] = "sup"
    return roles


__all__ = [
    "CONTROL_LINE_BREAK_CHARS",
    "ScriptCharFeature",
    "ScriptRole",
    "build_script_features",
    "classify_char_script_roles",
    "split_script_visual_components",
]
