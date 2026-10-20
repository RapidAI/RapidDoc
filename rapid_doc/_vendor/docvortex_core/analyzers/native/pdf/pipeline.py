"""编排 Flash 原生 PDF 的页面准备、语义处理和输出归一化。"""

from __future__ import annotations

from .typography import _font_family_page_scope

import re
import statistics

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from ....document.pdf._document import PDFDocument as PDFDocument
from ....document.pdf._document import PDFDrawingLine, PDFImageInfo, PDFPageTextGeometry
from ....schema import BBox
from .._shared.xycut import sort_entries
from ..contracts import NativePdfSource, RawBlock
from .auxiliary_text import (
    _classify_first_page_correspondence_footnotes,
    _classify_image_footnotes,
    _classify_page_footnotes,
    _build_marginal_candidate,
    _classify_deferred_image_footnotes,
    _classify_isolated_first_page_footer,
    _classify_page_auxiliary_text,
    _classify_page_footnote_trailing_footers,
    _classify_page_number_outer_companions,
    _classify_raw_page_marginals,
    _classify_repeated_page_marginals,
    _classify_repeated_visual_headers,
    _classify_rule_delimited_footers,
    _classify_rule_delimited_headers,
    _classify_single_page_compound_headers,
    _classify_split_marginal_row_companions,
    _marginal_geometry_matches,
    _marginal_text_matches,
)
from .char_geometry import DocumentGeometryPlan, apply_line_geometry_repairs, build_document_geometry_plan
from .code_blocks import _build_code_blocks, _build_rule_delimited_code_blocks
from .text_assembly.footnotes import _has_leading_footnote_superscript
from .text_assembly.continuity import group_native_inline_formula_prose
from .title_analysis.structural import (
    _classify_short_cjk_section_leads,
    _classify_bold_numbered_heading_rows,
    _mark_emphasized_quote_prose,
    _classify_native_display_resets,
    _restore_wrapped_bold_title_tails,
    _restore_short_heading_with_image_displaced_prose,
    _freeze_wrapped_bold_title_evidence,
)
from .formulas import (
    _formula_fresh_page_scope,
    classify_repeated_vector_decorations,
    _attach_unmapped_formula_ink,
    _build_formula_like_blocks,
    _native_math_word_fragments,
    _build_vector_formula_blocks,
    _unmapped_formula_ink_bboxes,
)
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_overlap_in_smaller,
    _bbox_overlap_in_first,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _normalize_bbox_to_unit,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
)
from .graphics import (
    _build_caption_graphic_blocks,
    _IMAGE_CONTAINER_OVERLAP_THRESHOLD,
    _build_form_image_blocks,
    _build_graphic_like_blocks,
    _build_raster_image_blocks,
    _detect_strong_graphic_bboxes,
    _form_supersedes_nested_bbox,
    _form_region_allows_line,
    _select_form_image_bboxes,
    _split_parallel_graphic_rule_rows,
)
from .index_blocks import _extract_index_blocks
from .inline.detection import detect_pdf_text_link_lines, detect_pdf_text_style_lines
from .inline.matching import _realign_repaired_text_evidence
from .inline.materialize import (
    apply_pdf_inline_evidence,
)
from .inline.scripts import _prepare_owned_script_evidence, detect_pdf_text_script_lines
from .inline.types import PDFTextLinkLine, PDFTextStyleLine
from .layout_evidence import build_layout_evidence
from .annotation_bands import recover_image_annotation_bands
from .title_analysis.compact_titles import _classify_small_emphasized_titles
from .text_roles import publication_text
from .line_merging import (
    _demote_runin_title_fragments,
    _merge_metric_heading_fragments,
    _merge_overlapping_inline_text_clusters,
    _merge_post_semantic_text_runs,
    _merge_same_baseline_text_lines,
    _merge_title_resolved_visual_rows,
    _restore_dense_split_visual_rows,
    merge_text_line_clusters,
)
from .models import (
    _AxisLine,
    _DocumentBodyProfile,
    _DocumentTitleProfile,
    _LineItem,
    _MarginalCandidate,
    _PageSource,
    _PreparedPage,
)
from .native_text import (
    _build_native_line_items_from_chars,
    _build_native_line_items_from_records,
    _coerce_pdf_drawing_lines,
    _extract_decorative_text_rules,
    _median_native_glyph_width,
    _resplit_native_visual_runs,
    _sanitize_pdf_control_text,
)
from .tables import _connected_horizontal_rule_bboxes, _detect_table_candidates, _materialize_table_blocks
from .text_assembly.annotations import (
    _merge_fragmented_header_blocks,
    _merge_front_matter_column_blocks,
    _merge_image_caption_text_blocks,
    _merge_multiline_title_blocks,
    _merge_repeated_compact_title_continuations,
)
from .text_assembly.assembly import _build_text_blocks, _restore_caption_wrap_text
from .text_assembly.annotations import _split_labeled_visual_note_blocks
from .text_assembly.common import _merge_internal_text_block_group
from .text_assembly.continuity import (
    group_front_matter_lines,
    group_reference_lines,
    mark_document_reference_regions,
    merge_overlapping_member_blocks,
    order_body_above_reference_band,
)
from .title_analysis.body_profile import _infer_document_body_profile
from .title_analysis.document_profile import _infer_document_title_profile
from .title_analysis.page_titles import _classify_page_titles
from .title_analysis.structural import (
    _demote_regular_repeated_item_titles,
    _classify_recurrent_unknown_weight_titles,
    _classify_body_height_section_titles,
    _classify_document_structural_titles,
    _classify_explicit_section_titles,
    _classify_inline_typography_reset_titles,
    _promote_noninitial_document_title_band,
)
from .visual_annotations import _classify_and_bind_visual_annotations, _is_strong_caption_text

_TEXT_SEMANTIC_TYPES = {
    "doc_title",
    "paragraph_title",
    "ref_text",
    "header",
    "footer",
    "page_number",
    "caption",
    "footnote",
    "page_footnote",
    "aside_text",
    "index",
}


_OUTPUT_BLOCK_TYPES = {"text", "table", "image", "equation", "code", "list"} | _TEXT_SEMANTIC_TYPES
_LINE_METADATA_OUTPUT_TYPES = {
    "text",
    "ref_text",
    "doc_title",
    "paragraph_title",
    "caption",
    "footnote",
}
_REPEATED_RASTER_IMAGE_MIN_PAGE_AREA_RATIO = 0.08
_REPEATED_RASTER_IMAGE_MIN_DISTINCT_PAGES = 3


def _is_large_raster_image(
    image_info: PDFImageInfo,
    page_size: tuple[float, float],
) -> bool:
    """判断点阵图裁剪后面积是否达到页面面积的 8%。"""

    bbox = _coerce_bbox(image_info.bbox)
    page_area = max(0.0, page_size[0]) * max(0.0, page_size[1])
    return bbox is not None and page_area > 0 and _bbox_area(bbox) / page_area >= _REPEATED_RASTER_IMAGE_MIN_PAGE_AREA_RATIO


def _detect_repeated_raster_watermark_fingerprints(
    page_image_infos: list[list[PDFImageInfo]],
    page_sizes: list[tuple[float, float]],
) -> set[str]:
    """按大图指纹统计不同页号，出现至少三页时判为跨页图片水印。"""

    page_indices_by_fingerprint: dict[str, set[int]] = {}
    for page_idx, (image_infos, page_size) in enumerate(zip(page_image_infos, page_sizes, strict=True)):
        for image_info in image_infos:
            if image_info.fingerprint is None or not _is_large_raster_image(image_info, page_size):
                continue
            page_indices_by_fingerprint.setdefault(image_info.fingerprint, set()).add(page_idx)
    return {
        fingerprint
        for fingerprint, page_indices in page_indices_by_fingerprint.items()
        if len(page_indices) >= _REPEATED_RASTER_IMAGE_MIN_DISTINCT_PAGES
    }


def _filter_repeated_raster_watermark_bboxes(
    image_infos: list[PDFImageInfo],
    page_size: tuple[float, float],
    watermark_fingerprints: set[str],
) -> list[tuple[float, float, float, float]]:
    """仅删除命中跨页水印指纹且面积达标的 bbox，小尺寸同图继续保留。"""

    return [
        image_info.bbox
        for image_info in image_infos
        if not (image_info.fingerprint in watermark_fingerprints and _is_large_raster_image(image_info, page_size))
    ]


def _table_detection_drawing_lines(
    source: _PageSource,
    confirmed_header_separators: set[BBox] | None = None,
) -> list[_AxisLine]:
    """从表格候选路径中移除已确认页眉下方的通栏分隔线。"""

    confirmed = confirmed_header_separators or set()
    return [drawing_line for drawing_line in source.drawing_lines if drawing_line.bbox not in confirmed]


def _detect_repeated_header_separator_bboxes(
    sources: list[_PageSource],
) -> list[set[BBox]]:
    """用重复刊头文本和非表格横线共同确认跨页页眉分隔线。"""

    candidates_by_signature: dict[
        tuple[float, float, float],
        list[tuple[int, BBox]],
    ] = {}
    for page_index, source in enumerate(sources):
        page_width, page_height = source.page_size
        if page_width <= 0 or page_height <= 0:
            continue
        connected_table_rules = _connected_horizontal_rule_bboxes(source)
        for drawing_line in source.drawing_lines:
            bbox = drawing_line.bbox
            center_y = _bbox_center_y(bbox)
            if (
                drawing_line.orientation != "horizontal"
                or center_y > 0.15 * page_height
                or bbox[2] - bbox[0] < 0.6 * page_width
                or sum(line.angle == 0 and line.bbox[3] <= center_y for line in source.lines) < 2
                or bbox in connected_table_rules
            ):
                continue
            signature = (
                round(bbox[0] / page_width, 2),
                round(bbox[2] / page_width, 2),
                round(center_y / page_height, 3),
            )
            candidates_by_signature.setdefault(
                signature,
                [],
            ).append((page_index, bbox))

    output = [set() for _source in sources]
    for members in candidates_by_signature.values():
        if len({page_index for page_index, _bbox in members}) < 3:
            continue
        if len(_repeated_header_evidence_pages(sources, members)) < 2:
            continue
        for page_index, bbox in members:
            output[page_index].add(bbox)
    return output


def _repeated_header_evidence_pages(
    sources: list[_PageSource],
    separator_members: list[tuple[int, BBox]],
) -> set[int]:
    """返回具有已分类或跨页重复刊头文本证据的页号集合。"""

    supported_pages: set[int] = set()
    candidates: list[tuple[int, _MarginalCandidate]] = []
    for page_index, separator_bbox in separator_members:
        center_y = _bbox_center_y(separator_bbox)
        for line in sources[page_index].lines:
            if line.angle != 0 or line.bbox[3] > center_y:
                continue
            if line.semantic_type == "header":
                supported_pages.add(page_index)
                continue
            candidate = _build_marginal_candidate(
                page_index,
                line,
                sources[page_index].page_size,
            )
            if candidate is not None and candidate.region == "header":
                candidates.append((page_index, candidate))

    for left_index, (left_page, left) in enumerate(candidates):
        for right_page, right in candidates[left_index + 1 :]:
            page_delta = right_page - left_page
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and _marginal_geometry_matches(left, right)
                and _marginal_text_matches(left.line.text, right.line.text)
            ):
                supported_pages.update((left_page, right_page))
    return supported_pages


_STANDARD_STYLE_DETECTOR = detect_pdf_text_style_lines


@dataclass(slots=True)
class _DocumentSources:
    """持有跨页校准前的原始页面，以及最终物化仍需的紧凑样式证据。"""

    page_sources: list[_PageSource]
    page_text_geometries: list[PDFPageTextGeometry]
    page_sizes: list[tuple[float, float]]
    page_style_lines: list[list[PDFTextStyleLine]]
    page_link_lines: list[list[PDFTextLinkLine]]
    page_owned_scripts: list[Any] = field(default_factory=list)
    page_geometry_evidence: list[Any] = field(default_factory=list)
    page_spacing_lines: list[list[Any]] = field(default_factory=list)


def _prepare_owned_geometry_evidence(owner, chars):
    """一次准备全文风险所需的页面级 Rust 几何记录和原字符身份表。"""
    if not hasattr(owner, "prepare_geometry_evidence"):
        return None
    evidence = owner.prepare_geometry_evidence()
    if evidence is None:
        return None
    return evidence, {id(char): index for index, char in enumerate(chars)}


def _collect_document_sources(pdf_doc: NativePdfSource) -> _DocumentSources:
    """逐页收集原生证据，局部快照与字符引用在收集阶段退出时释放。"""

    from .inline.spacing import prepare_spacing_lines

    page_sizes: list[tuple[float, float]] = []
    page_image_infos: list[list[PDFImageInfo]] = []
    page_sources: list[_PageSource] = []
    page_text_geometries = []
    page_style_lines: list[list[PDFTextStyleLine]] = []
    page_link_lines: list[list[PDFTextLinkLine]] = []
    page_owned_scripts = []
    page_geometry_evidence = []
    page_spacing_lines = []
    for page_idx in range(pdf_doc.page_count):
        snapshot = pdf_doc._extract_native_page(page_idx)
        page_size = snapshot.page_size
        page_sizes.append(page_size)
        page_image_infos.append(snapshot.image_infos)
        native_text = getattr(snapshot, "native_text", None)
        if native_text is not None:
            text_geometry, records = native_text.prepare_visual_evidence(page_size, snapshot.rotation, (0.0, 90.0, 270.0))
            lines = _build_native_line_items_from_records(records, page_size)
        else:
            text_geometry = snapshot.text_geometry
            lines = _build_native_line_items_from_chars(
                text_geometry.chars,
                page_size,
                page_rotation=snapshot.rotation,
            )
        from .local_text import group_hanging_letter_paragraphs, restore_drop_cap_paragraphs

        restore_drop_cap_paragraphs(lines)
        group_hanging_letter_paragraphs(lines)
        space_before = set(native_text.tight_space_indices()) if native_text is not None else None
        page_spacing_lines.append(prepare_spacing_lines(lines, space_before))
        chars = text_geometry.chars
        # 只保留纯 Rust 脚本/几何记录与本页身份映射，不延长 PDFium 页面或文档句柄生命周期。
        page_owned_scripts.append(_prepare_owned_script_evidence(native_text, chars) if native_text is not None else None)
        page_geometry_evidence.append(_prepare_owned_geometry_evidence(native_text, chars) if native_text is not None else None)
        drawing_lines = _coerce_pdf_drawing_lines(snapshot.drawing_lines)
        raster_em = sorted(line.em_height for line in lines if line.em_height > 0)
        if raster_em:
            em = raster_em[len(raster_em) // 2]
            # 原生 PDF 可用一像素图片画分隔线；仅转投极薄水平图的几何，不识别其文字。
            drawing_lines.extend(
                _coerce_pdf_drawing_lines(
                    [
                        PDFDrawingLine(
                            (info.bbox[0], (info.bbox[1] + info.bbox[3]) / 2),
                            (info.bbox[2], (info.bbox[1] + info.bbox[3]) / 2),
                            info.bbox,
                            info.bbox[3] - info.bbox[1],
                            "horizontal",
                        )
                        for info in snapshot.image_infos
                        if info.bbox[2] - info.bbox[0] >= 2 * em and 0 < info.bbox[3] - info.bbox[1] <= 0.1 * em
                    ]
                )
            )
        lines, decorative_rules = _extract_decorative_text_rules(
            lines,
            page_size,
        )
        drawing_lines.extend(decorative_rules)
        from .inline.owned_styles import detect_owned_style_lines

        owned = page_owned_scripts[-1]
        style_lines = (
            detect_owned_style_lines(native_text, lines, drawing_lines, owned[1] if owned is not None else None)
            if detect_pdf_text_style_lines is _STANDARD_STYLE_DETECTOR
            else None
        )
        page_style_lines.append(style_lines if style_lines is not None else detect_pdf_text_style_lines(lines, drawing_lines))
        from .inline.glyph_weight import glyph_weight_style_lines

        render_page = getattr(pdf_doc, "render_page", None)
        page_style_lines[-1].extend(
            glyph_weight_style_lines(
                lines,
                text_geometry,
                (lambda: render_page(page_idx, scale=3.0)) if callable(render_page) else None,
                owner=native_text,
                identities=owned[1] if owned is not None else None,
            )
        )
        page_link_lines.append(
            detect_pdf_text_link_lines(
                lines,
                snapshot.link_annotations,
            )
        )
        source = _PageSource(
            page_size=page_size,
            page_index=page_idx,
            lines=lines,
            chars=chars,
            drawing_lines=drawing_lines,
            signature_bboxes=snapshot.signature_bboxes,
            form_bboxes=snapshot.form_bboxes,
            path_infos=snapshot.path_infos,
            form_infos=getattr(snapshot, "form_infos", ()),
        )
        page_sources.append(source)
        page_text_geometries.append(text_geometry)

    watermark_fingerprints = _detect_repeated_raster_watermark_fingerprints(page_image_infos, page_sizes)
    for source, image_infos in zip(page_sources, page_image_infos, strict=True):
        if len(source.lines) >= 3 and sum(len(line.text.strip()) for line in source.lines) >= 60:
            # 低纹理整页图只是底色，原生文字和内容图片仍由独立规则处理；无原生文字的图页保留。
            image_infos = [info for info in image_infos if not info.smooth_background]
        source.image_bboxes = _filter_repeated_raster_watermark_bboxes(
            image_infos,
            source.page_size,
            watermark_fingerprints,
        )
        source.image_bboxes = _exclude_proven_blank_top_under_display_title(source, image_infos)

    from .text_noise import exclude_faint_native_noise
    from .graphics import group_native_raster_chart_descriptions

    paint_reader = getattr(pdf_doc, "_get_page_text_paint", None)
    for source in page_sources:
        exclude_faint_native_noise(
            source,
            (lambda indices, page=source.page_index: paint_reader(page, indices)) if callable(paint_reader) else None,
            _detect_strong_graphic_bboxes(source),
        )
        group_native_raster_chart_descriptions(source)

    return _DocumentSources(
        page_sources,
        page_text_geometries,
        page_sizes,
        page_style_lines,
        page_link_lines,
        page_owned_scripts,
        page_geometry_evidence,
        page_spacing_lines,
    )


def _exclude_proven_blank_top_under_display_title(source, image_infos):
    """仅当大字号文字与图像自身白边相交时裁去空白顶边，保留原生图片内容及标题独立性。"""
    heights = [line.effective_height for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not heights:
        return source.image_bboxes
    body = statistics.median(heights)
    evidence = {info.bbox: getattr(info, "blank_top_bbox", None) for info in image_infos}
    result = []
    for box in source.image_bboxes:
        trimmed = evidence.get(box)
        if trimmed is not None and any(
            line.angle == 0
            and line.effective_height >= 1.4 * body
            and (line.dominant_font_weight or 400) >= 600
            and _bbox_axis_overlap_ratio(line.bbox, box, axis="x") >= 0.5
            and line.bbox[1] < trimmed[1]
            and box[1] + 0.1 * body < line.bbox[3] <= trimmed[1]
            for line in source.lines
        ):
            result.append(trimmed)
        else:
            result.append(box)
    return result


def _prepare_document_sources(
    sources: _DocumentSources,
    *,
    geometry_diagnostics: list[dict[str, Any]] | None = None,
) -> list[_PreparedPage]:
    """先完成全文几何和页眉判定，再按顺序消费原始页面并释放已用证据。"""

    geometry_plan = build_document_geometry_plan(
        [source.lines for source in sources.page_sources],
        sources.page_text_geometries,
        sources.page_sizes,
        owned_geometry_inputs=sources.page_geometry_evidence,
    )
    for page_index, source in enumerate(sources.page_sources):
        # 容器认领前只允许 X 修复，表格和图形认领后再启用 Y trim。
        apply_line_geometry_repairs(source.lines, page_index=page_index, plan=geometry_plan, allow_y_trim=False)
    if geometry_diagnostics is not None:
        geometry_diagnostics.append(geometry_plan.to_dict())
    _classify_raw_page_marginals(sources.page_sources)
    from .form_roles import normalize_page_forms

    normalize_page_forms(sources.page_sources)
    separators = _detect_repeated_header_separator_bboxes(sources.page_sources)
    repaired_chars_by_page: dict[int, dict[int, BBox]] = {}
    for (page_index, char_idx), repair in geometry_plan.char_repairs.items():
        repaired_chars_by_page.setdefault(page_index, {})[char_idx] = repair.layout_bbox

    owned_scripts = sources.page_owned_scripts or [None] * len(sources.page_sources)
    pending = deque(zip(sources.page_sources, sources.page_text_geometries, owned_scripts, strict=True))
    sources.page_owned_scripts.clear()
    sources.page_geometry_evidence.clear()
    del owned_scripts
    sources.page_sources.clear()
    sources.page_text_geometries.clear()
    prepared_pages: list[_PreparedPage] = []
    from .table_rules import _trailing_table_continuation_grid

    previous_table_grid = None
    while pending:
        page_index = len(prepared_pages)
        source, geometry, owned_script = pending.popleft()
        source.previous_table_grid = previous_table_grid
        prepared_pages.append(
            _prepare_page_source(
                source,
                tight_bboxes=geometry.tight_bboxes,
                origins=geometry.origins,
                geometry_plan=geometry_plan,
                page_index=page_index,
                style_lines=sources.page_style_lines[page_index],
                link_lines=sources.page_link_lines[page_index],
                table_header_separator_bboxes=separators[page_index],
                repaired_char_bboxes=repaired_chars_by_page.pop(page_index, {}),
                _owned_script_inputs=owned_script,
            )
        )
        previous_table_grid = _trailing_table_continuation_grid(source, prepared_pages[-1])
        # 删除对象所有者引用，不清空共享字符容器，公式重建副本仍可安全使用。
        del source, geometry, owned_script
    return prepared_pages


@dataclass(frozen=True, slots=True)
class _DocumentTextProfiles:
    """分别保存原始正文尺度、规范正文尺度及全文标题原型。"""

    body: _DocumentBodyProfile | None
    canonical_body: _DocumentBodyProfile | None
    title: _DocumentTitleProfile | None


def _classify_document_text(prepared_pages: list[_PreparedPage]) -> _DocumentTextProfiles:
    """按既有顺序分类跨页辅助文本，再统计正文并确认结构标题。"""

    _classify_repeated_visual_headers(prepared_pages)
    _classify_repeated_page_marginals(prepared_pages)
    _classify_split_marginal_row_companions(prepared_pages)
    _classify_single_page_compound_headers(prepared_pages)
    _classify_rule_delimited_headers(prepared_pages)
    _classify_rule_delimited_footers(prepared_pages)
    _classify_page_number_outer_companions(prepared_pages)
    _classify_page_footnote_trailing_footers(prepared_pages)
    _classify_isolated_first_page_footer(prepared_pages)
    _restore_table_preamble_metadata(prepared_pages)
    _restore_background_body_marginals(prepared_pages)
    document_body_profile = _infer_document_body_profile(prepared_pages)
    canonical_body_profile = _infer_document_body_profile(
        prepared_pages,
        use_canonical_scale=True,
    )
    if document_body_profile is not None:
        _classify_deferred_image_footnotes(
            prepared_pages,
            document_body_profile.body_height,
        )
    document_title_profile = _infer_document_title_profile(
        prepared_pages,
        document_body_profile,
    )
    _classify_document_structural_titles(
        prepared_pages,
        canonical_body_profile,
        legacy_body_profile=document_body_profile,
        document_title_profile=document_title_profile,
    )
    _classify_recurrent_unknown_weight_titles(prepared_pages)
    return _DocumentTextProfiles(document_body_profile, canonical_body_profile, document_title_profile)


def _materialize_document_inline(
    finalized_pages: list[list[dict[str, Any]]],
    prepared_pages: list[_PreparedPage],
    sources: _DocumentSources,
    script_diagnostics: list[dict[str, Any]] | None,
) -> None:
    """在页面归一化后按链接、样式、上下标顺序物化最终行内语义。"""

    from .inline.spacing import apply_spacing_lines

    for page_index, (page_blocks, prepared, style_lines, link_lines, page_size) in enumerate(
        zip(
            finalized_pages,
            prepared_pages,
            sources.page_style_lines,
            sources.page_link_lines,
            sources.page_sizes,
            strict=True,
        )
    ):
        materialized_diagnostics = None
        if script_diagnostics is not None:
            materialized_diagnostics = []
            script_diagnostics[page_index]["materialized_ranges"] = materialized_diagnostics
        if sources.page_spacing_lines:
            apply_spacing_lines(page_blocks, sources.page_spacing_lines[page_index], page_size)
        apply_pdf_inline_evidence(
            page_blocks,
            link_lines,
            style_lines,
            prepared.script_lines,
            page_size,
            materialized_diagnostics=materialized_diagnostics,
        )


def _analyze_native_document(
    pdf_doc: NativePdfSource,
    *,
    script_diagnostics: list[dict[str, Any]] | None = None,
    geometry_diagnostics: list[dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """逐页读取数字 PDF，并在轻量页面上完成跨页文本类型判定。"""

    sources = _collect_document_sources(pdf_doc)
    prepared_pages = _prepare_document_sources(sources, geometry_diagnostics=geometry_diagnostics)
    if script_diagnostics is not None:
        script_diagnostics.extend(
            {
                "page_index": page_index,
                "page_size": prepared.page_size,
                "script_lines": list(prepared.script_lines),
                "lines": [
                    {
                        "source_index": line.source_index,
                        "text": line.text,
                        "bbox": line.bbox,
                        "angle": line.angle,
                    }
                    for line in prepared.remaining_lines
                ],
            }
            for page_index, prepared in enumerate(prepared_pages)
        )

    classify_repeated_vector_decorations(prepared_pages)
    profiles = _classify_document_text(prepared_pages)
    mark_document_reference_regions(prepared_pages)
    for page_index, prepared in enumerate(prepared_pages):
        if prepared.numbered_references:
            group_reference_lines(prepared.remaining_lines, prepared)
        _classify_small_emphasized_titles(prepared, page_index)
    finalized_pages = [
        _finalize_prepared_page(
            prepared,
            page_index,
            canonical_body_profile=profiles.canonical_body,
            document_body_profile=profiles.body,
            document_title_profile=profiles.title,
        )
        for page_index, prepared in enumerate(prepared_pages)
    ]
    _materialize_document_inline(finalized_pages, prepared_pages, sources, script_diagnostics)
    from .table_media import materialize_table_media

    materialize_table_media(finalized_pages, pdf_doc)
    return finalized_pages


def _build_caption_supported_graphics(source: _PageSource) -> tuple[list[dict[str, Any]], set[int]]:
    """在流水线汇总图题、表格和真实代码屏障，图形领域不反向依赖其他领域分类器。"""
    from .table_detection import _detect_table_candidates, has_native_closed_grid

    caption_indices = {line.source_index for line in source.lines if _is_strong_caption_text(line.text)}
    if not any(
        line.source_index in caption_indices and re.match(r"^\s*(?:figure|fig\.)", line.text, re.IGNORECASE)
        for line in source.lines
    ):
        return [], set()
    table_caption = any(re.match(r"^\s*(?:table|tab\.?|表)\s*\d", line.text, re.IGNORECASE) for line in source.lines)
    table_bboxes = [
        candidate.core_bbox or candidate.bbox
        for candidate in _detect_table_candidates(source)
        if (table_caption and candidate.annotations) or has_native_closed_grid(source, candidate)
    ]
    code_bboxes = [block["bbox"] for block in _build_code_blocks(source, [], set())[0]]
    code_bboxes.extend(
        block["bbox"]
        for block in _build_rule_delimited_code_blocks(source, [], set())[0]
        if len(
            re.findall(r"(?im)^\s*(?:\d+\s*[:.]?\s*)?(?:int|float|void|class|def|for|while|if|return|else)\b", block["content"])
        )
        >= 2
    )
    return _build_caption_graphic_blocks(
        source, caption_line_indices=caption_indices, table_bboxes=table_bboxes, code_bboxes=code_bboxes
    )


def _mark_native_caption_starts(lines: list[_LineItem]) -> None:
    """保存图题首行身份以阻止向左吞并邻栏正文，完整图题仍由后续绑定流程分类。"""
    for line in lines:
        if line.angle == 0 and _is_strong_caption_text(line.text):
            line.caption_start = True


def _restore_background_body_marginals(pages: list[_PreparedPage]) -> None:
    """Unique native foreground on a page background is not a repeated marginal."""
    frequencies = {}
    for page in pages:
        for value in {re.sub(r'\s+', '', line.text) for line in page.remaining_lines}:
            frequencies[value] = frequencies.get(value, 0) + 1
    for page in pages:
        backgrounds = [block['bbox'] for block in page.fixed_blocks if block['type'] == 'image'
                       and _bbox_area(block['bbox']) >= .6 * page.page_size[0] * page.page_size[1]]
        for line in page.remaining_lines:
            if (line.semantic_type in {'header', 'footer', 'aside_text'} and len(line.text.strip()) > 3
                    and frequencies.get(re.sub(r'\s+', '', line.text)) == 1
                    and any(_bbox_overlap_in_first(line.bbox, image) >= .95 for image in backgrounds)):
                line.semantic_type = None
                line.marginal_body_context = True


def _restore_table_preamble_metadata(pages: list[_PreparedPage]) -> None:
    """Retain the connected title/metadata band belonging to each page's table."""
    for page in pages:
        for table in page.table_bboxes:
            members = [line for line in page.remaining_lines if line.angle == 0
                       and line.semantic_type in {None, 'header', 'aside_text'}
                       and len(line.text.strip()) > 3
                       and 0 <= table[1] - line.bbox[3] <= 4 * max(.1, line.effective_height)
                       and line.bbox[0] < table[2] and line.bbox[2] > table[0]
                       and not any(rule.orientation == "horizontal"
                                   and line.bbox[3] < _bbox_center_y(rule.bbox) < table[1] - .25 * max(.1, line.effective_height)
                                   and min(rule.bbox[2], table[2]) - max(rule.bbox[0], table[0]) >= .6 * (table[2] - table[0])
                                   for rule in page.drawing_lines)]
            # Walk upward from the table; a detached running header cannot borrow
            # the lower metadata group's width and height to become body text.
            connected = []
            top = table[1]
            next_em = None
            for line in sorted(members, key=lambda line: line.bbox[3], reverse=True):
                em = max(.1, line.effective_height)
                if top - line.bbox[3] > 1.5 * max(em, next_em or em):
                    break
                connected.append(line)
                top = min(top, line.bbox[1])
                next_em = em
            members = connected
            if len(members) < 2:
                continue
            bounds = _bbox_union_many([line.bbox for line in members])
            em = statistics.median(max(.1, line.effective_height) for line in members)
            if bounds[3] - bounds[1] < 1.5 * em or bounds[2] - bounds[0] < .7 * (table[2] - table[0]):
                continue
            for line in members:
                line.semantic_type = None
                line.marginal_body_context = True


@_font_family_page_scope
def _prepare_page_source(
    source: _PageSource,
    *,
    tight_bboxes: dict[int, BBox] | None = None,
    origins: dict[int, tuple[float, float]] | None = None,
    geometry_plan: DocumentGeometryPlan | None = None,
    page_index: int = 0,
    style_lines: list[PDFTextStyleLine] | None = None,
    link_lines: list[PDFTextLinkLine] | None = None,
    table_header_separator_bboxes: set[BBox] | None = None,
    repaired_char_bboxes: dict[int, BBox] | None = None,
    _owned_script_inputs=None,
) -> _PreparedPage:
    """先认领视觉容器，再标注辅助文本并留下可跨页比较的轻量文本行。"""

    if page_index == 0:
        _classify_first_page_correspondence_footnotes(source)
    # 在表格认领前保留有明确目录题名的稳定页码列，避免目录横线被误当数据网格。
    early_index_blocks, _ = _extract_index_blocks(
        source.lines,
        source.page_size,
        source.image_bboxes + source.signature_bboxes,
        require_heading=True,
        require_explicit_heading=True,
    )
    early_index_indices = {line.source_index for line in source.lines if line.semantic_type == "index"}
    caption_graphics, caption_graphic_claims = _build_caption_supported_graphics(source)
    caption_graphic_bboxes = [block["bbox"] for block in caption_graphics]
    _classify_bold_numbered_heading_rows(
        source.lines,
        source.page_size,
        source.image_bboxes + caption_graphic_bboxes + _detect_strong_graphic_bboxes(source),
        appendix_only=True,
    )
    protected_line_indices = {line.source_index for line in source.lines if line.semantic_type is not None}
    analysis_source = replace(
        source,
        lines=[line for line in source.lines if line.source_index not in protected_line_indices | caption_graphic_claims],
        publication_bboxes=[line.bbox for line in source.lines if publication_text(line.text)],
    )
    form_bboxes = [
        bbox
        for bbox in _select_form_image_bboxes(source)
        if not any(_bbox_overlap_in_smaller(bbox, graphic) >= 0.8 for graphic in caption_graphic_bboxes)
    ]
    strong_graphic_bboxes = _detect_strong_graphic_bboxes(analysis_source) + caption_graphic_bboxes
    # 原生数值表的多列表头与重复金额列共同证明容器，不能被轴线外观抢先认领为图形。
    from .table_detection import _detect_aligned_numeric_tables

    numeric_table_evidence = _detect_aligned_numeric_tables(analysis_source, list(source.image_bboxes))
    strong_graphic_bboxes = [
        box
        for box in strong_graphic_bboxes
        if not any(
            _bbox_overlap_in_smaller(box, candidate.bbox) >= 0.5 and _bbox_area(box) >= 0.25 * _bbox_area(candidate.bbox)
            for candidate in numeric_table_evidence
        )
    ]
    rule_code_blocks, claimed_rule_code_line_indices = _build_rule_delimited_code_blocks(
        analysis_source,
        form_bboxes + strong_graphic_bboxes + list(source.image_bboxes) + list(source.signature_bboxes),
    )
    rule_code_bboxes = [block["bbox"] for block in rule_code_blocks]
    table_analysis_source = replace(
        analysis_source,
        drawing_lines=_table_detection_drawing_lines(
            source,
            table_header_separator_bboxes,
        ),
    )
    candidates = [
        candidate
        for candidate in _detect_table_candidates(
            table_analysis_source,
            excluded_bboxes=strong_graphic_bboxes + rule_code_bboxes,
        )
        if not any(_form_supersedes_nested_bbox(form_bbox, candidate.bbox) for form_bbox in form_bboxes)
    ]
    # 候选检测仍避开预分类边缘文本；已确认表格物化时回到原始行，
    # 让 core_bbox 内的误标页脚可被重新认领，表格外边缘文本不会被矩形扩张带入。
    table_blocks, table_annotation_blocks, claimed_line_indices = _materialize_table_blocks(
        source,
        candidates,
        tight_bboxes=tight_bboxes,
        origins=origins,
        calibrated_char_bboxes={
            char_idx: repair.layout_bbox
            for (repair_page, char_idx), repair in geometry_plan.char_repairs.items()
            if repair_page == page_index and repair.y_state == "abnormal"
        }
        if geometry_plan is not None and geometry_plan.local_metric_repair_sources
        else None,
        _owned_script_inputs=_owned_script_inputs,
    )
    from .graphics import _graphic_members_are_numeric_grid

    for block in table_blocks:
        members = [line for line in source.lines if _bbox_overlap_in_first(line.bbox, block["bbox"]) >= 0.95]
        block["_native_numeric_grid"] = _graphic_members_are_numeric_grid(members)
    claimed_line_indices.update(caption_graphic_claims)
    claimed_line_indices.update(early_index_indices)
    claimed_line_indices.update(claimed_rule_code_line_indices)
    table_bboxes = [block["bbox"] for block in table_blocks]
    active_form_bboxes = [
        form_bbox
        for form_bbox in form_bboxes
        if not any(
            _bbox_overlap_in_smaller(form_bbox, table_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD for table_bbox in table_bboxes
        )
    ]
    code_blocks, claimed_code_line_indices = _build_code_blocks(
        analysis_source,
        table_bboxes + active_form_bboxes + strong_graphic_bboxes + list(source.image_bboxes) + list(source.signature_bboxes),
        claimed_line_indices,
    )
    code_bboxes = [block["bbox"] for block in code_blocks]
    form_image_blocks, claimed_form_line_indices = _build_form_image_blocks(
        analysis_source,
        active_form_bboxes,
        claimed_line_indices | claimed_code_line_indices,
    )
    graphic_blocks, claimed_graphic_line_indices = _build_graphic_like_blocks(
        analysis_source,
        table_bboxes + active_form_bboxes + rule_code_bboxes + code_bboxes + caption_graphic_bboxes,
        claimed_line_indices | claimed_code_line_indices | claimed_form_line_indices,
        strong_graphic_bboxes,
    )
    raster_image_blocks, claimed_raster_line_indices = _build_raster_image_blocks(
        analysis_source,
        table_blocks + rule_code_blocks + code_blocks + form_image_blocks + graphic_blocks + caption_graphics,
        claimed_line_indices | claimed_code_line_indices | claimed_form_line_indices | claimed_graphic_line_indices,
    )
    vector_formula_blocks, claimed_vector_number_indices = _build_vector_formula_blocks(
        analysis_source,
        table_blocks
        + rule_code_blocks
        + code_blocks
        + form_image_blocks
        + graphic_blocks
        + raster_image_blocks
        + caption_graphics,
        claimed_line_indices
        | claimed_code_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices,
    )
    claimed_line_indices = (
        claimed_line_indices
        | claimed_code_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices
        | claimed_vector_number_indices
    )
    unclaimed_lines = [line for line in source.lines if line.source_index not in claimed_line_indices]
    if geometry_plan is not None:
        apply_line_geometry_repairs(
            unclaimed_lines,
            page_index=page_index,
            plan=geometry_plan,
            allow_y_trim=True,
        )
    next_source_index = (
        max(
            (line.source_index for line in source.lines),
            default=-1,
        )
        + 1
    )
    if geometry_plan is not None:
        repaired_line_bboxes = {line.source_index: line.bbox for line in source.lines}
        if repaired_char_bboxes is None:
            # 单页内部入口保留独立调用能力；文档主链路已提前按页建立索引。
            repaired_char_bboxes = {
                char_idx: repair.layout_bbox
                for (repair_page_index, char_idx), repair in geometry_plan.char_repairs.items()
                if repair_page_index == page_index
            }
        unclaimed_lines, resplits = _resplit_native_visual_runs(
            unclaimed_lines,
            source.page_size,
            repaired_char_bboxes,
            source_index_start=next_source_index,
        )
        aligned_styles, aligned_links = _realign_repaired_text_evidence(
            style_lines or [],
            link_lines or [],
            repaired_line_bboxes,
            resplits,
        )
        if style_lines is not None and aligned_styles is not style_lines:
            style_lines[:] = aligned_styles
        if link_lines is not None and aligned_links is not link_lines:
            link_lines[:] = aligned_links
        next_source_index = max(
            next_source_index,
            max(
                (line.source_index for line in unclaimed_lines),
                default=-1,
            )
            + 1,
        )
    remaining_lines = _split_parallel_graphic_rule_rows(
        unclaimed_lines,
        source.drawing_lines,
        [block["bbox"] for block in (form_image_blocks + graphic_blocks + raster_image_blocks)],
        table_bboxes,
        source.page_size,
        source_index_start=next_source_index,
    )
    # 跨栏原生 run 在 X 校准后才能正确拆开，补认领这时才暴露出的图内标签。
    # 所有已确认内容图共享成员补认领：几何校准后才进入点阵图/矢量图的原生标签也不能重复输出。
    image_containers = caption_graphics + form_image_blocks + graphic_blocks + raster_image_blocks
    reclaimed = set()
    reclaimed_by_owner: dict[int, list[_LineItem]] = {}
    original_image_content = {id(block): block["content"] for block in image_containers}
    for line in remaining_lines:
        if line.semantic_type is not None or _is_strong_caption_text(line.text):
            continue
        ink = line.ink_bbox or line.bbox
        matches = [
            block
            for block in image_containers
            if _bbox_overlap_in_first(ink, block["bbox"]) >= 0.98
            and not (
                _bbox_area(block["bbox"]) >= .6 * source.page_size[0] * source.page_size[1]
                and any(_bbox_overlap_in_first(block["bbox"], raster) >= .95 for raster in source.image_bboxes)
            )
            and (
                _form_region_allows_line(source, line, block["bbox"])
                # 柱图等强绘图证据确认整体容器，允许由外部文本对象叠加的刻度/图例归入该图。
                or sum(
                    _bbox_area(core) for core in strong_graphic_bboxes if _bbox_overlap_in_first(core, block["bbox"]) >= 0.95
                )
                >= 0.25 * _bbox_area(block["bbox"])
            )
        ]
        if matches:
            owner = min(matches, key=lambda block: _bbox_area(block["bbox"]))
            reclaimed_by_owner.setdefault(id(owner), []).append(line)
            reclaimed.add(line.source_index)
    from .graphics import _image_members_to_content

    for owner in image_containers:
        members = reclaimed_by_owner.get(id(owner), [])
        if members:
            # 整批补认领的图内标签须按原生位置投影，不能沿 PDF 对象序逐条追加。
            projected = _image_members_to_content(members, source.page_size, spatial=True)
            owner["content"] = "\n".join(filter(None, [original_image_content[id(owner)], projected]))
    remaining_lines = [line for line in remaining_lines if line.source_index not in reclaimed]
    from .formulas import restore_leading_inline_fractions

    remaining_lines = restore_leading_inline_fractions(remaining_lines, source.drawing_lines, source.page_size)
    canonical_formula_source_lines = (
        [
            replace(
                line,
                chars=list(line.chars),
                inline_math_regions=list(line.inline_math_regions),
            )
            for line in remaining_lines
        ]
        if geometry_plan is not None and geometry_plan.document_style_anomaly
        else []
    )
    early_visual_bboxes = [
        block["bbox"] for block in caption_graphics + form_image_blocks + graphic_blocks + raster_image_blocks
    ]
    _mark_native_caption_starts(remaining_lines)
    from .title_analysis.native_boundaries import classify_bilingual_term_bands, classify_appendix_title_bands

    classify_bilingual_term_bands(remaining_lines)
    classify_appendix_title_bands(remaining_lines, source.page_size)
    _mark_emphasized_quote_prose(remaining_lines)
    group_native_inline_formula_prose(
        remaining_lines,
        [
            (block["bbox"], block["_inline_formula_prose_sources"])
            for block in vector_formula_blocks
            if block.get("_inline_formula_prose_sources")
        ],
    )
    _classify_native_display_resets(
        remaining_lines,
        source.page_size,
        early_visual_bboxes,
        table_bboxes,
        page_index=page_index,
        reference_lines=source.lines,
    )
    from .native_cjk_bands import restore_native_cjk_bands

    restore_native_cjk_bands(remaining_lines, source.page_size, early_visual_bboxes + table_bboxes, page_index)
    _classify_image_footnotes(remaining_lines, early_visual_bboxes, table_bboxes, source.drawing_lines, source.page_size)
    from .title_analysis.native_boundaries import protect_parallel_table_end_fields

    protect_parallel_table_end_fields(remaining_lines, table_bboxes)
    early_footnote_groups = _classify_page_footnotes(
        remaining_lines,
        table_bboxes,
        source.drawing_lines,
        source.page_size,
        visual_bboxes=early_visual_bboxes,
        reference_lines=source.lines,
    )
    remaining_lines = merge_text_line_clusters(remaining_lines, source.page_size, table_bboxes)
    formula_candidate_lines = [line for line in remaining_lines if line.formula_candidate_only]
    remaining_lines = [line for line in remaining_lines if not line.formula_candidate_only]
    script_lines = detect_pdf_text_script_lines(
        remaining_lines,
        source.page_size,
        tight_bboxes or {},
        origins or {},
        all_chars=source.chars,
        drawing_lines=source.drawing_lines,
        _owned_inputs=_owned_script_inputs,
    )
    _compact_prepared_lines(remaining_lines, source.page_size)
    _compact_prepared_lines(formula_candidate_lines, source.page_size)
    prepared = _PreparedPage(
        page_size=source.page_size,
        remaining_lines=remaining_lines,
        table_bboxes=table_bboxes,
        drawing_lines=source.drawing_lines,
        fixed_blocks=(
            rule_code_blocks
            + table_annotation_blocks
            + table_blocks
            + code_blocks
            + form_image_blocks
            + graphic_blocks
            + raster_image_blocks
            + vector_formula_blocks
            + caption_graphics
            + early_index_blocks
        ),
        canonical_formula_source_lines=canonical_formula_source_lines,
        script_lines=script_lines,
        formula_candidate_lines=formula_candidate_lines,
        formula_ink_bboxes=_unmapped_formula_ink_bboxes(source.chars),
        table_body_profile_lines=(
            [
                replace(line, chars=[], inline_math_regions=[])
                for line in source.lines
                if any(_bbox_overlap_in_first(line.bbox, bounds) >= 0.9 for bounds in table_bboxes)
            ]
            if len(remaining_lines) <= 6 and table_bboxes
            else []
        ),
        page_footnote_groups=early_footnote_groups,
        bullet_paths=tuple(
            info
            for info in source.path_infos
            if info.fill_visible and not info.stroke_visible and not info.rectangle_bboxes and 8 <= info.segment_count <= 100
        ),
    )
    _classify_page_auxiliary_text(prepared)
    return prepared


def _compact_prepared_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """缓存后续仍需的字符尺度并释放字符字典，限制跨页阶段内存占用。"""

    for line in lines:
        cjk = [char for char in line.chars or [] if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", ""))]
        if cjk:
            font = cjk[0].get("font") or {}
            line.native_cjk_style = (
                str(font.get("name", "")),
                int(font.get("flags") or 0),
                float(font.get("weight") or 0),
                statistics.median(char["bbox"][3] - char["bbox"][1] for char in cjk),
            )
        if line.median_glyph_width is None:
            line.median_glyph_width = _median_native_glyph_width(line, page_size)
        # 释放字符前冻结脚注首标的几何证据，供后续分项使用。
        line.footnote_marker_start = line.footnote_marker_start or _has_leading_footnote_superscript(line)
        line.native_math_words = _native_math_word_fragments(line)
        _freeze_wrapped_bold_title_evidence(line, page_size)
        line.chars.clear()


def _rebuild_canonical_formula_blocks(
    prepared: _PreparedPage,
    excluded_source_indices: set[int],
) -> list[dict[str, Any]]:
    """从容器认领后的未合并行重放 canonical 公式路径，避免 loose 行高改变公式成员顺序。"""

    replay_lines = [
        replace(
            line,
            chars=list(line.chars),
            style_scale_repaired=True,
            inline_math_regions=list(line.inline_math_regions),
        )
        for line in prepared.canonical_formula_source_lines
        if line.source_index not in excluded_source_indices
    ]
    replay_lines = _merge_same_baseline_text_lines(
        replay_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    replay_lines = _merge_overlapping_inline_text_clusters(
        replay_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    replay_lines = _merge_same_baseline_text_lines(
        replay_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_candidate_lines = [line for line in replay_lines if line.formula_candidate_only]
    replay_lines = [line for line in replay_lines if not line.formula_candidate_only]
    formula_input = _restore_dense_split_visual_rows(
        replay_lines + formula_candidate_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_input = _merge_same_baseline_text_lines(
        formula_input,
        prepared.page_size,
        prepared.table_bboxes,
    )
    blocks, _remaining_lines = _build_formula_like_blocks(
        formula_input,
        prepared.table_bboxes,
        prepared.page_size,
        drawing_lines=prepared.drawing_lines,
    )
    return blocks


def _formula_block_inventory(
    blocks: list[dict[str, Any]],
) -> list[tuple[float, float, float, float]]:
    """返回公式重放与正常路径可比较的稳定 bbox 库存。"""

    return sorted(
        tuple(float(value) for value in block["bbox"])
        for block in blocks
        if block.get("type") == "equation" and isinstance(block.get("bbox"), (list, tuple)) and len(block["bbox"]) == 4
    )


def _apply_post_aggregation_tight_bboxes(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> None:
    """在 block 聚合完成后应用 tight+1pt 框，并同步最终公开行框。"""

    for block in blocks:
        candidate_bbox = _coerce_bbox(
            block.pop("_tight_output_bbox", None),
        )
        output_line_bboxes = block.pop(
            "_local_output_line_bboxes",
            None,
        )
        output_bbox_repaired = block.pop(
            "_output_bbox_repaired",
            False,
        )
        if output_bbox_repaired is True and isinstance(output_line_bboxes, list) and output_line_bboxes:
            local_bboxes = [_coerce_bbox(value) for value in output_line_bboxes]
            if all(value is not None for value in local_bboxes):
                resolved_local_bboxes = [value for value in local_bboxes if value is not None]
                block["_local_line_bboxes"] = resolved_local_bboxes
                angle = int(block.get("angle", 0) or 0) % 360
                candidate_bbox = _bbox_union_many(
                    [
                        _rotate_bbox_from_upright(
                            value,
                            page_size,
                            angle,
                        )
                        for value in resolved_local_bboxes
                    ]
                )
        clipped_bbox = _clip_bbox(candidate_bbox, page_size)
        if clipped_bbox is not None:
            block["bbox"] = clipped_bbox


@_font_family_page_scope
@_formula_fresh_page_scope
def _finalize_prepared_page(
    prepared: _PreparedPage,
    page_index: int,
    *,
    canonical_body_profile: _DocumentBodyProfile | None = None,
    document_body_profile: _DocumentBodyProfile | None = None,
    document_title_profile: _DocumentTitleProfile | None = None,
) -> list[dict[str, Any]]:
    """按预分类语义、公式、标题、正文的优先级完成单页文本并排序。"""

    semantic_lines = [line for line in prepared.remaining_lines if line.semantic_type is not None]
    unresolved_lines = [line for line in prepared.remaining_lines if line.semantic_type is None]
    container_bboxes = [block["bbox"] for block in prepared.fixed_blocks]
    index_blocks, unresolved_lines = _extract_index_blocks(
        unresolved_lines,
        prepared.page_size,
        container_bboxes,
        require_heading=True,
    )
    semantic_lines.extend(line for line in unresolved_lines if line.semantic_type is not None)
    formula_input = [
        *(line for line in unresolved_lines if line.semantic_type is None),
        *prepared.formula_candidate_lines,
    ]
    original_style_scale_state = {line.source_index: line.style_scale_repaired for line in formula_input}
    if prepared.canonical_formula_geometry:
        for line in formula_input:
            line.style_scale_repaired = True
    formula_input = _restore_dense_split_visual_rows(
        formula_input,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_input = _merge_same_baseline_text_lines(
        formula_input,
        prepared.page_size,
        prepared.table_bboxes,
    )
    formula_blocks, remaining_lines = _build_formula_like_blocks(
        formula_input,
        prepared.table_bboxes,
        prepared.page_size,
        drawing_lines=prepared.drawing_lines,
    )
    if prepared.canonical_formula_geometry and prepared.canonical_formula_source_lines:
        canonical_formula_blocks = _rebuild_canonical_formula_blocks(
            prepared,
            {line.source_index for line in semantic_lines},
        )
        if _formula_block_inventory(
            canonical_formula_blocks,
        ) == _formula_block_inventory(formula_blocks):
            formula_blocks = canonical_formula_blocks
    prepared.canonical_formula_source_lines.clear()
    _attach_unmapped_formula_ink(formula_blocks, prepared.formula_ink_bboxes)
    if prepared.canonical_formula_geometry:
        for line in remaining_lines:
            line.style_scale_repaired = original_style_scale_state.get(
                line.source_index,
                line.style_scale_repaired,
            )
    fallback_index_blocks, remaining_lines = _extract_index_blocks(
        remaining_lines,
        prepared.page_size,
        container_bboxes,
    )
    index_blocks.extend(fallback_index_blocks)
    semantic_lines.extend(line for line in remaining_lines if line.semantic_type is not None)
    remaining_lines = [line for line in remaining_lines if line.semantic_type is None]
    title_container_bboxes = [
        block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
    ]
    if page_index == 0:
        group_front_matter_lines(remaining_lines, prepared.page_size)
    caption_container_bboxes = [block["bbox"] for block in prepared.fixed_blocks if block.get("type") in {"image", "code"}]
    _classify_explicit_section_titles(
        remaining_lines,
        prepared.page_size,
        container_bboxes=title_container_bboxes,
        document_body_profile=document_body_profile,
    )
    _classify_inline_typography_reset_titles(
        remaining_lines,
        prepared.page_size,
        container_bboxes=title_container_bboxes,
        document_body_profile=document_body_profile,
    )
    _classify_body_height_section_titles(
        remaining_lines,
        prepared.page_size,
        container_bboxes=title_container_bboxes,
        document_body_profile=document_body_profile,
        page_index=page_index,
    )
    _classify_page_titles(
        remaining_lines,
        prepared.page_size,
        page_index=page_index,
        container_bboxes=title_container_bboxes,
        caption_container_bboxes=caption_container_bboxes,
        document_body_profile=document_body_profile,
        document_title_profile=document_title_profile,
    )
    from .title_analysis.native_boundaries import suppress_continuous_sentence_titles

    suppress_continuous_sentence_titles(remaining_lines)
    _promote_noninitial_document_title_band(
        remaining_lines,
        prepared.page_size,
        page_index=page_index,
        container_bboxes=title_container_bboxes,
        document_body_profile=(canonical_body_profile or document_body_profile),
        title_candidate_source_indices={
            line.source_index for line in remaining_lines if line.semantic_type == "paragraph_title"
        },
    )
    remaining_lines = semantic_lines + remaining_lines
    semantic_lines = []
    if page_index == 0:
        for line in remaining_lines:
            if line.text.strip().casefold() in {"abstract", "摘要"}:
                line.semantic_type = "paragraph_title"
    _demote_runin_title_fragments(remaining_lines, prepared.page_size)
    _demote_regular_repeated_item_titles(remaining_lines, document_body_profile)
    from .title_analysis.panels import classify_panel_titles

    classify_panel_titles(remaining_lines, caption_container_bboxes, prepared.page_size[0])
    _classify_bold_numbered_heading_rows(remaining_lines, prepared.page_size, title_container_bboxes)
    remaining_lines = _restore_wrapped_bold_title_tails(remaining_lines, prepared.page_size)
    _restore_short_heading_with_image_displaced_prose(
        remaining_lines, [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"]
    )
    _classify_short_cjk_section_leads(remaining_lines)
    remaining_lines = _merge_metric_heading_fragments(remaining_lines, prepared.page_size)
    remaining_lines = _merge_title_resolved_visual_rows(
        remaining_lines,
        prepared.page_size,
    )
    remaining_lines = _merge_post_semantic_text_runs(
        remaining_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    group_reference_lines(remaining_lines, prepared)
    if page_index == 0:
        group_front_matter_lines(remaining_lines, prepared.page_size)
    text_blocks = _build_text_blocks(
        semantic_lines + remaining_lines,
        prepared.table_bboxes,
        prepared.page_size,
        prepared.drawing_lines,
        page_footnote_groups=prepared.page_footnote_groups,
        page_index=page_index,
        visual_bboxes=[block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
    )
    text_blocks = merge_overlapping_member_blocks(text_blocks, prepared.page_size)
    text_blocks = _merge_multiline_title_blocks(text_blocks)
    text_blocks = _merge_front_matter_column_blocks(
        text_blocks,
        prepared.page_size,
        page_index=page_index,
    )
    text_blocks = _merge_image_caption_text_blocks(
        text_blocks,
        [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
        [
            block["bbox"]
            for block in prepared.fixed_blocks
            if block.get("type") == "table" and block.get("_native_numeric_grid")
        ],
    )
    text_blocks = _merge_fragmented_header_blocks(text_blocks)
    text_blocks = _restore_caption_wrap_text(
        text_blocks,
        [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
        prepared.page_size,
    )
    text_blocks = _merge_repeated_compact_title_continuations(
        text_blocks,
        prepared.page_size,
    )
    text_blocks = _split_labeled_visual_note_blocks(
        text_blocks, [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"]
    )
    from .local_lists import split_local_list_blocks

    text_blocks = split_local_list_blocks(text_blocks, prepared.bullet_paths)
    absolute_blocks = prepared.fixed_blocks + formula_blocks + index_blocks + text_blocks
    recover_image_annotation_bands(absolute_blocks, prepared.page_size, prepared.drawing_lines)
    _apply_post_aggregation_tight_bboxes(
        absolute_blocks,
        prepared.page_size,
    )
    visual_annotation_regions = _classify_and_bind_visual_annotations(
        absolute_blocks,
        prepared.page_size,
        merge_text_block_group=_merge_internal_text_block_group,
    )
    sorted_blocks = _sort_blocks_with_visual_row_groups(
        absolute_blocks,
        prepared.page_size,
        visual_annotation_regions=visual_annotation_regions,
    )
    sorted_blocks = _preserve_narrow_graphic_column_order(sorted_blocks, prepared.page_size)
    sorted_blocks = order_body_above_reference_band(sorted_blocks, prepared)
    # 首页机构编号按物理行排列，避免长机构行使 XY-cut 将短机构先后顺序交错。
    front_matter = iter(
        sorted(
            (block for block in sorted_blocks if (block.get("_reference_group") or 0) < 0),
            key=lambda block: block["bbox"][1],
        )
    )
    sorted_blocks = [next(front_matter) if (block.get("_reference_group") or 0) < 0 else block for block in sorted_blocks]
    return [
        normalized for block in sorted_blocks if (normalized := _normalize_output_block(block, prepared.page_size)) is not None
    ]


def _preserve_narrow_graphic_column_order(blocks: list[dict[str, Any]], page_size: tuple[float, float]) -> list[dict[str, Any]]:
    """同栏子图合并不应把另一栏的图表插进连续正文；跨栏页面继续使用原排序。"""
    if not any(block.get("_caption_graphic") for block in blocks):
        return blocks
    width = page_size[0]
    marginals = {"header", "footer", "page_number", "aside_text"}
    content = [block for block in blocks if block["type"] not in marginals]
    body = [block for block in content if block["type"] == "text" and len(block.get("_text_lines", [])) >= 2]
    layout = build_layout_evidence([line for block in body for line in block["_text_lines"]], page_size)
    if len(layout.lanes) < 2 or len(body) < 3:
        return blocks
    corridors = {id(block): layout.corridor(block["bbox"]) for block in content}
    if any(c is None or c == (0.0, width) for c in corridors.values()):
        return blocks
    ordered = iter(sorted(content, key=lambda block: (corridors[id(block)][0], block["bbox"][1], block["bbox"][0])))
    return [block if block["type"] in marginals else next(ordered) for block in blocks]


def _analyze_page_source(source: _PageSource) -> list[dict[str, Any]]:
    """兼容单页测试入口；单页不凭边缘位置猜测页眉、页脚或页码。"""

    if not source.lines and not source.image_bboxes and not source.signature_bboxes:
        return []
    return _finalize_prepared_page(_prepare_page_source(source), page_index=0)


def _numbered_panel_row_regions(
    regions: list[list[dict[str, Any]]], body_blocks: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """连续图号及重复行列位置共同证明逐行面板网格；夹有正文或编号按栏排列时保留原栏序。"""
    panels = []
    for region in regions:
        captions = [member for member in region if member.get("type") == "caption"]
        bodies = [member for member in region if member.get("type") == "image"]
        if len(captions) != 1 or len(bodies) != 1:
            continue
        caption = captions[0]
        match = re.match(r"^\s*(?:图|Figure\s*|Fig\.?\s*)(\d+)(?:\s|[:：.])", str(caption.get("content", "")), re.I)
        if match is None or caption["bbox"][1] > bodies[0]["bbox"][1]:
            continue
        panels.append((region, caption, int(match.group(1))))
    if len(panels) < 4:
        return regions
    em = statistics.median(panel[1]["bbox"][3] - panel[1]["bbox"][1] for panel in panels)
    if em <= 0:
        return regions
    rows = []
    for panel in sorted(panels, key=lambda item: (item[1]["bbox"][1], item[1]["bbox"][0])):
        if not rows or abs(panel[1]["bbox"][1] - rows[-1][0][1]["bbox"][1]) > 0.75 * em:
            rows.append([])
        rows[-1].append(panel)
    if len(rows) < 2 or not all(2 <= len(row) == len(rows[0]) <= 3 for row in rows):
        return regions
    rows = [sorted(row, key=lambda item: item[1]["bbox"][0]) for row in rows]
    ordered = [panel for row in rows for panel in row]
    if [panel[2] for panel in ordered] != list(range(ordered[0][2], ordered[0][2] + len(ordered))):
        return regions
    if any(
        abs(panel[1]["bbox"][0] - rows[0][column][1]["bbox"][0]) > 0.75 * em
        for row in rows[1:]
        for column, panel in enumerate(row)
    ):
        return regions
    members = [member for panel in panels for member in panel[0]]
    member_ids = {id(member) for member in members}
    bounds = _bbox_union_many([member["bbox"] for member in members])
    if any(
        id(block) not in member_ids
        and block.get("type") in {"text", "paragraph_title", "doc_title", "table", "equation", "code"}
        and _bbox_axis_overlap_ratio(block["bbox"], bounds, axis="x") >= 0.1
        and _bbox_axis_overlap_ratio(block["bbox"], bounds, axis="y") >= 0.1
        for block in body_blocks
    ):
        return regions
    selected = {id(panel[0]) for panel in panels}
    return [
        *[region for region in regions if id(region) not in selected],
        *[[member for panel in row for member in panel[0]] for row in rows],
    ]


def _sort_blocks_with_visual_row_groups(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    visual_annotation_regions: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """把视觉注释区域和拆分粗行包装成虚拟项排序，再按各自局部顺序展开。"""

    top_marginals: list[dict[str, Any]] = []
    bottom_marginals: list[dict[str, Any]] = []
    body_blocks: list[dict[str, Any]] = []
    local_page_height = page_size[1]
    for block in blocks:
        block_type = block.get("type")
        bbox = block.get("bbox")
        if block_type == "header" or (
            block_type == "page_number" and isinstance(bbox, (list, tuple)) and _bbox_center_y(bbox) <= 0.5 * local_page_height
        ):
            top_marginals.append(block)
        elif block_type in {"footer", "page_footnote"} or block_type == "page_number":
            bottom_marginals.append(block)
        else:
            body_blocks.append(block)

    body_index_by_identity = {id(block): index for index, block in enumerate(body_blocks)}
    region_groups: list[list[dict[str, Any]]] = []
    region_consumed_indices: set[int] = set()
    for region in visual_annotation_regions or []:
        indices = [body_index_by_identity[id(member)] for member in region if id(member) in body_index_by_identity]
        if len(indices) < 2 or any(index in region_consumed_indices for index in indices):
            continue
        members = [body_blocks[index] for index in indices]
        for member in members:
            member["_visual_annotation_region_member"] = True
        region_groups.append(members)
        region_consumed_indices.update(indices)

    region_groups = _numbered_panel_row_regions(region_groups, body_blocks)
    from .panel_order import numbered_step_member_groups, parallel_heading_panel_groups

    for group in parallel_heading_panel_groups(body_blocks, region_consumed_indices):
        region_groups.append([member for _, member in group])
        region_consumed_indices.update(index for index, _ in group)
    for group in numbered_step_member_groups(body_blocks, region_consumed_indices):
        region_groups.append([member for _, member in group])
        region_consumed_indices.update(index for index, _ in group)

    inline_grouped_indices = _collect_inline_image_text_groups(
        body_blocks,
        excluded_indices=region_consumed_indices,
    )
    inline_consumed_indices = {index for indices in inline_grouped_indices.values() for index in indices}
    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(body_blocks):
        if index in region_consumed_indices or index in inline_consumed_indices:
            continue
        row_id = block.get("_single_run_row_id")
        if isinstance(row_id, int):
            grouped_indices.setdefault(row_id, []).append(index)

    virtual_groups: list[dict[str, Any]] = []
    consumed_indices: set[int] = set(region_consumed_indices | inline_consumed_indices)
    body_layout = build_layout_evidence(
        [
            line
            for block in body_blocks
            if block.get("type") == "text" and len(block.get("_text_lines", [])) >= 3
            for line in block["_text_lines"]
        ],
        page_size,
    )
    for members in region_groups:
        region_bbox = _bbox_union_many([member["bbox"] for member in members])
        sorting_bbox = region_bbox
        if len(body_layout.lanes) == 1:
            lane = body_layout.lanes[0]
            # 单栏独立小图占据整段阅读流；只扩展排序投影，真实裁图及成员边界不变。
            # 同高度仍有环绕正文时继续按局部几何切栏。
            if lane.left <= region_bbox[0] < region_bbox[2] <= lane.right and not any(
                other not in members
                and other.get("type") in {"text", "paragraph_title", "image", "table", "code"}
                and _bbox_axis_overlap_ratio(other["bbox"], region_bbox, axis="y") >= 0.1
                for other in body_blocks
            ):
                sorting_bbox = (lane.left, region_bbox[1], lane.right, region_bbox[3])
        virtual_groups.append(
            {
                "type": "_xycut_visual_annotation_region",
                "bbox": sorting_bbox,
                "angle": members[0].get("angle", 0),
                "content": "",
                "_members": members,
                "_visual_annotation_region": True,
            }
        )
    for row_id, indices in inline_grouped_indices.items():
        members = [body_blocks[index] for index in indices]
        virtual_groups.append(
            {
                "type": "_xycut_visual_row_group",
                "bbox": _bbox_union_many([member["bbox"] for member in members]),
                "angle": members[0].get("angle", 0),
                "content": "",
                "_members": members,
                "_inline_visual_row_id": row_id,
            }
        )
    for row_id, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        members = [body_blocks[index] for index in indices]
        virtual_group = {
            "type": "_xycut_visual_row_group",
            "bbox": _bbox_union_many([member["bbox"] for member in members]),
            "angle": members[0].get("angle", 0),
            "content": "",
            "_members": members,
        }
        virtual_groups.append(virtual_group)
        consumed_indices.update(indices)

    sortable_blocks = [block for index, block in enumerate(body_blocks) if index not in consumed_indices]
    sortable_blocks.extend(virtual_groups)
    sorted_payloads = sort_entries(sortable_blocks)
    output: list[dict[str, Any]] = []
    for payload in sorted_payloads:
        members = payload.get("_members")
        if not isinstance(members, list):
            output.append(payload)
            continue
        if payload.get("_visual_annotation_region") is True:
            output.extend(members)
            continue
        if isinstance(payload.get("_inline_visual_row_id"), int):
            members.sort(
                key=lambda member: _inline_visual_group_member_sort_key(
                    member,
                    page_size,
                )
            )
            output.extend(members)
            continue
        angle = int(payload.get("angle", 0) or 0) % 360
        members.sort(
            key=lambda member: (
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[0],
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[1],
            )
        )
        output.extend(members)
    output = _stabilize_overlapping_lane_order(output, page_size)
    from .title_analysis.native_boundaries import order_leading_title_bands

    output = order_leading_title_bands(output, page_size)
    from .formulas import order_equation_markers

    output = order_equation_markers(output)
    return [
        *_sort_marginal_blocks(top_marginals, page_size),
        *output,
        *_sort_marginal_blocks(bottom_marginals, page_size),
    ]


def _collect_inline_image_text_groups(
    body_blocks: list[dict[str, Any]],
    *,
    excluded_indices: set[int] | None = None,
) -> dict[int, list[int]]:
    """把复合图片与包含同一首行的正文块组成专用排序组。"""

    excluded_indices = excluded_indices or set()
    image_indices_by_row: dict[int, list[int]] = {}
    for index, block in enumerate(body_blocks):
        if index in excluded_indices:
            continue
        row_id = block.get("_inline_visual_row_id")
        if block.get("type") == "image" and isinstance(row_id, int):
            image_indices_by_row.setdefault(row_id, []).append(index)

    output: dict[int, list[int]] = {}
    consumed_text_indices: set[int] = set()
    for row_id, image_indices in sorted(image_indices_by_row.items()):
        text_indices = [
            index
            for index, block in enumerate(body_blocks)
            if index not in excluded_indices
            and index not in consumed_text_indices
            and block.get("type") == "text"
            and isinstance(block.get("_visual_row_ids"), set)
            and row_id in block["_visual_row_ids"]
        ]
        if not text_indices:
            continue
        output[row_id] = [*image_indices, *text_indices]
        consumed_text_indices.update(text_indices)
    return output


def _inline_visual_group_member_sort_key(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> tuple[float, float, int]:
    """按图片位置或正文首个局部行位置确定复合视觉行的组内顺序。"""

    local_line_bboxes = block.get("_local_line_bboxes")
    if block.get("type") == "text" and isinstance(local_line_bboxes, list) and local_line_bboxes:
        local_bbox = local_line_bboxes[0]
    else:
        angle = int(block.get("angle", 0) or 0) % 360
        local_bbox = _rotate_bbox_to_upright(
            block["bbox"],
            page_size,
            angle,
        )
    return (
        float(local_bbox[0]),
        float(local_bbox[1]),
        0 if block.get("type") == "image" else 1,
    )


def _sort_marginal_blocks(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """先按视觉中心聚合边缘同排块，再按行内 x 排序以消除字体框顶边抖动。"""

    notes = [block for block in blocks if block.get("type") == "page_footnote"]
    if len(notes) >= 2 and len(notes) == len(blocks):
        # 脚注是连续条目，沿不相交栏带读完一栏再换栏；页眉/页脚仍保持同排从左到右。
        return sort_entries(notes)
    if notes and len(notes) < len(blocks):
        ordered_notes = iter(_sort_marginal_blocks(notes, page_size))
        ordinary = _sort_marginal_blocks([block for block in blocks if block.get("type") != "page_footnote"], page_size)
        return [*ordered_notes, *ordinary]
    if len(blocks) < 2:
        return list(blocks)
    geometry = [
        (
            block,
            _rotate_bbox_to_upright(
                block["bbox"],
                page_size,
                int(block.get("angle", 0) or 0) % 360,
            ),
        )
        for block in blocks
    ]
    heights = [
        float(height)
        for block in blocks
        for height in block.get("_line_heights", [])
        if isinstance(height, (int, float)) and height > 0
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 1.0
    rows: list[list[tuple[dict[str, Any], tuple[float, float, float, float]]]] = []
    for item in sorted(geometry, key=lambda value: _bbox_center_y(value[1])):
        target = next(
            (
                row
                for row in rows
                if abs(_bbox_center_y(item[1]) - sum(_bbox_center_y(member[1]) for member in row) / len(row))
                <= 0.75 * median_height
            ),
            None,
        )
        if target is None:
            rows.append([item])
        else:
            target.append(item)
    rows.sort(key=lambda row: sum(_bbox_center_y(member[1]) for member in row) / len(row))
    return [block for row in rows for block, _bbox in sorted(row, key=lambda member: member[1][0])]


def _stabilize_overlapping_lane_order(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """对同栏轻微重叠块按视觉中心纠正局部逆序，不改变跨栏主阅读顺序。"""

    output = list(blocks)
    for _pass_index in range(len(output)):
        changed = False
        for index in range(len(output) - 1):
            first = output[index]
            second = output[index + 1]
            if not _overlapping_lane_pair_is_inverted(first, second, page_size):
                continue
            output[index], output[index + 1] = second, first
            changed = True
        if not changed:
            break
    return output


def _overlapping_lane_pair_is_inverted(
    first: dict[str, Any],
    second: dict[str, Any],
    page_size: tuple[float, float],
) -> bool:
    """判断相邻块是否属于同一内部栏带且视觉中心顺序与当前结果相反。"""

    if first.get("_visual_annotation_region_member") or second.get("_visual_annotation_region_member"):
        return False
    first_interval = first.get("_lane_interval")
    second_interval = second.get("_lane_interval")
    if (
        not isinstance(first_interval, (list, tuple))
        or not isinstance(second_interval, (list, tuple))
        or len(first_interval) != 2
        or len(second_interval) != 2
        or first.get("_lane_is_span") != second.get("_lane_is_span")
        or int(first.get("angle", 0) or 0) % 360 != int(second.get("angle", 0) or 0) % 360
    ):
        return False
    first_bbox = _rotate_bbox_to_upright(
        first["bbox"],
        page_size,
        int(first.get("angle", 0) or 0) % 360,
    )
    second_bbox = _rotate_bbox_to_upright(
        second["bbox"],
        page_size,
        int(second.get("angle", 0) or 0) % 360,
    )
    line_heights = [
        float(height)
        for block in (first, second)
        for height in block.get("_line_heights", [])
        if isinstance(height, (int, float)) and height > 0
    ]
    tolerance = 0.75 * (min(line_heights) if line_heights else 1.0)
    same_lane = (
        abs(float(first_interval[0]) - float(second_interval[0])) <= tolerance
        and abs(float(first_interval[1]) - float(second_interval[1])) <= tolerance
    )
    vertical_overlap = min(first_bbox[3], second_bbox[3]) - max(
        first_bbox[1],
        second_bbox[1],
    )
    return (
        same_lane
        and _bbox_center_y(first_bbox) > _bbox_center_y(second_bbox) + 0.1 * tolerance
        and _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x") >= 0.35
        and vertical_overlap >= 0.0
    )


def _normalize_output_block(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """在排序完成后将绝对 bbox 裁剪并归一化为 model_list 坐标。"""

    page_width, page_height = page_size
    bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None
    content = block.get("content")
    if not isinstance(content, str):
        return None
    content = _sanitize_pdf_control_text(content, preserve_newlines=True)
    block_type = block.get("type")
    normalized_type = block_type if block_type in _OUTPUT_BLOCK_TYPES else "text"
    if normalized_type not in {"image", "equation", "header", "footer", "list"} and not content.strip():
        return None
    normalized_bbox = _normalize_bbox_to_unit(bbox, page_size)
    output_block: RawBlock = {
        "type": normalized_type,
        "bbox": normalized_bbox,
        "angle": 0 if normalized_type == "image" else int(block.get("angle", 0) or 0) % 360,
        "content": "" if normalized_type == "equation" else content,
    }
    inline_math_regions = []
    for value in block.get("_inline_math_regions", []):
        raw_region = _coerce_bbox(value)
        if raw_region is None:
            continue
        region = _clip_bbox(raw_region, page_size)
        if region is not None:
            inline_math_regions.append(_normalize_bbox_to_unit(region, page_size))
    if inline_math_regions:
        output_block["_inline_math_regions"] = inline_math_regions
    if block.get("_table_images"):
        output_block["_table_images"] = [
            {**image, "bbox": _normalize_bbox_to_unit(image["bbox"], page_size)}
            for image in block["_table_images"]
        ]
    if normalized_type in _LINE_METADATA_OUTPUT_TYPES:
        output_block["lines"] = _normalize_output_line_items(block, page_size)
    members = block.get("_text_lines", [])
    if members and members[0].reference_start is not None:
        output_block["_reference_start"] = members[0].reference_start
    if block.get("_protected_hard_break_before"):
        output_block["_paragraph_boundary"] = True
    parent_bbox = _coerce_bbox(block.get("_native_annotation_parent_bbox"))
    if parent_bbox is not None:
        output_block["_native_annotation_parent_bbox"] = _normalize_bbox_to_unit(parent_bbox, page_size)
    return output_block


def _normalize_output_line_items(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> list[dict[str, list[float]]]:
    """将 Flash 正向局部行框逆变换为页面坐标并归一化输出。"""

    local_line_bboxes = block.get("_local_line_bboxes")
    if not isinstance(local_line_bboxes, list):
        return []

    angle = int(block.get("angle", 0) or 0) % 360
    line_items: list[dict[str, list[float]]] = []
    for local_line_bbox in local_line_bboxes:
        try:
            raw_bbox = tuple(float(value) for value in local_line_bbox)
        except (TypeError, ValueError):
            return []
        coerced_bbox = _coerce_bbox(raw_bbox)
        if len(raw_bbox) != 4 or coerced_bbox is None or raw_bbox[2] <= raw_bbox[0] or raw_bbox[3] <= raw_bbox[1]:
            return []
        page_bbox = _clip_bbox(
            _rotate_bbox_from_upright(coerced_bbox, page_size, angle),
            page_size,
        )
        if page_bbox is None:
            return []
        line_items.append({"bbox": _normalize_bbox_to_unit(page_bbox, page_size)})
    return line_items
