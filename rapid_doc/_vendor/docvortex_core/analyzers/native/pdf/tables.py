"""PDF 表格的兼容入口；实现按检测、注释和物化职责组织。"""

from ....document.pdf._document import PDFPathInfo as PDFPathInfo
from ....document.pdf.text._contracts import Char as Char
from ....foundation._text import merge_text_line_contents as merge_text_line_contents
from ....schema import (
    BBox as BBox,
)
from ._table_recovery import NativeTableInput as NativeTableInput
from ._table_recovery import NativeTableRectangle as NativeTableRectangle
from ._table_recovery import NativeTableRule as NativeTableRule
from ._table_recovery import coerce_native_table_rectangles as coerce_native_table_rectangles
from ._table_recovery import coerce_native_table_rules as coerce_native_table_rules
from ._table_recovery import recover_native_pdf_table as recover_native_pdf_table
from .geometry import (
    _bbox_area as _bbox_area,
)
from .geometry import (
    _bbox_axis_overlap_ratio as _bbox_axis_overlap_ratio,
)
from .geometry import (
    _bbox_center_x as _bbox_center_x,
)
from .geometry import (
    _bbox_center_y as _bbox_center_y,
)
from .geometry import (
    _bbox_overlap_in_smaller as _bbox_overlap_in_smaller,
)
from .geometry import (
    _bbox_union as _bbox_union,
)
from .geometry import (
    _bbox_union_many as _bbox_union_many,
)
from .geometry import (
    _coerce_bbox as _coerce_bbox,
)
from .geometry import (
    _expand_bbox as _expand_bbox,
)
from .geometry import (
    _point_in_bbox as _point_in_bbox,
)
from .geometry import (
    _rotate_bbox_from_upright as _rotate_bbox_from_upright,
)
from .geometry import (
    _rotate_bbox_to_upright as _rotate_bbox_to_upright,
)
from .geometry import (
    _transform_axis_lines as _transform_axis_lines,
)
from .line_layout import (
    _font_signatures_share_family as _font_signatures_share_family,
)
from .line_layout import (
    _line_effective_height as _line_effective_height,
)
from .line_layout import (
    _line_tight_output_bbox as _line_tight_output_bbox,
)
from .line_merging import (
    _same_baseline_geometry as _same_baseline_geometry,
)
from .models import (
    _Fragment as _Fragment,
)
from .models import (
    _LineItem as _LineItem,
)
from .models import (
    _LocalAxisLine as _LocalAxisLine,
)
from .models import (
    _PageSource as _PageSource,
)
from .models import (
    _TableAnnotation as _TableAnnotation,
)
from .models import (
    _TableCandidate as _TableCandidate,
)
from .models import (
    _VisualRow as _VisualRow,
)
from .native_text import (
    _normalize_native_run_text as _normalize_native_run_text,
)
from .spatial_text import (
    project_pdf_table_text as project_pdf_table_text,
)
from .table_annotations import (
    _build_table_annotation as _build_table_annotation,
)
from .table_annotations import (
    _collect_caption_rows as _collect_caption_rows,
)
from .table_annotations import (
    _collect_footnote_rows as _collect_footnote_rows,
)
from .table_annotations import (
    _extract_auxiliary_table_note_marker as _extract_auxiliary_table_note_marker,
)
from .table_annotations import (
    _find_caption_number_peers as _find_caption_number_peers,
)
from .table_annotations import (
    _find_table_caption as _find_table_caption,
)
from .table_annotations import (
    _is_table_note_text as _is_table_note_text,
)
from .table_annotations import (
    _line_has_compact_marker_token as _line_has_compact_marker_token,
)
from .table_annotations import (
    _line_has_superscript_marker as _line_has_superscript_marker,
)
from .table_annotations import (
    _merge_table_candidate_annotations as _merge_table_candidate_annotations,
)
from .table_annotations import (
    _table_core_references_marker as _table_core_references_marker,
)
from .table_annotations import (
    _table_note_body_reference_height as _table_note_body_reference_height,
)
from .table_annotations import (
    _visual_row_text as _visual_row_text,
)
from .table_constants import (
    _AUXILIARY_TABLE_NOTE_RE as _AUXILIARY_TABLE_NOTE_RE,
)
from .table_constants import (
    _FILLED_GRID_MAX_PAGE_AREA_RATIO as _FILLED_GRID_MAX_PAGE_AREA_RATIO,
)
from .table_constants import (
    _FILLED_GRID_MIN_PAGE_AREA_RATIO as _FILLED_GRID_MIN_PAGE_AREA_RATIO,
)
from .table_constants import (
    _FILLED_GRID_MIN_PAGE_HEIGHT_RATIO as _FILLED_GRID_MIN_PAGE_HEIGHT_RATIO,
)
from .table_constants import (
    _FILLED_GRID_MIN_PAGE_WIDTH_RATIO as _FILLED_GRID_MIN_PAGE_WIDTH_RATIO,
)
from .table_constants import (
    _TABLE_CAPTION_RE as _TABLE_CAPTION_RE,
)
from .table_constants import (
    _TABLE_CONTINUATION_RE as _TABLE_CONTINUATION_RE,
)
from .table_constants import (
    _TABLE_NOTE_RE as _TABLE_NOTE_RE,
)
from .table_constants import (
    _TABLE_SPLIT_NUMBER_RE as _TABLE_SPLIT_NUMBER_RE,
)
from .table_detection import (
    _detect_table_candidates as _detect_table_candidates,
)
from .table_detection import (
    _externalize_table_continuation_captions as _externalize_table_continuation_captions,
)
from .table_filled_grid import (
    _bbox_is_contained_with_tolerance as _bbox_is_contained_with_tolerance,
)
from .table_filled_grid import (
    _detect_filled_grid_table_candidates as _detect_filled_grid_table_candidates,
)
from .table_filled_grid import (
    _filled_grid_band_covers_outer_width as _filled_grid_band_covers_outer_width,
)
from .table_filled_grid import (
    _group_filled_grid_cells_into_bands as _group_filled_grid_cells_into_bands,
)
from .table_filled_grid import (
    _select_maximal_filled_grid_cells as _select_maximal_filled_grid_cells,
)
from .table_materialization import (
    _build_table_annotation_blocks as _build_table_annotation_blocks,
)
from .table_materialization import (
    _candidate_projection_line_indices as _candidate_projection_line_indices,
)
from .table_materialization import (
    _expand_candidate_same_baseline_members as _expand_candidate_same_baseline_members,
)
from .table_materialization import (
    _materialize_table_annotations as _materialize_table_annotations,
)
from .table_materialization import (
    _materialize_table_blocks as _materialize_table_blocks,
)
from .table_materialization import (
    _merge_table_annotation_content as _merge_table_annotation_content,
)
from .table_materialization import (
    _recover_native_table_html as _recover_native_table_html,
)
from .table_materialization import (
    _split_table_annotation_visual_groups as _split_table_annotation_visual_groups,
)
from .table_materialization import (
    _table_body_materialization_bbox as _table_body_materialization_bbox,
)
from .table_rows import (
    _clip_visual_row_to_corridor as _clip_visual_row_to_corridor,
)
from .table_rules import (
    _build_closed_rule_grid_candidates as _build_closed_rule_grid_candidates,
)
from .table_rules import (
    _build_fragments as _build_fragments,
)
from .table_rules import (
    _build_rule_table_candidates as _build_rule_table_candidates,
)
from .table_rules import (
    _closed_grid_vertical_track_positions as _closed_grid_vertical_track_positions,
)
from .table_rules import (
    _cluster_fragment_rows as _cluster_fragment_rows,
)
from .table_rules import (
    _compact_fully_ruled_grid_column_count as _compact_fully_ruled_grid_column_count,
)
from .table_rules import (
    _connected_horizontal_rule_bboxes as _connected_horizontal_rule_bboxes,
)
from .table_rules import (
    _connected_rule_grid_bboxes as _connected_rule_grid_bboxes,
)
from .table_rules import (
    _connected_rule_grid_components as _connected_rule_grid_components,
)
from .table_rules import (
    _continuous_table_row_segments as _continuous_table_row_segments,
)
from .table_rules import (
    _count_aligned_vertical_rules as _count_aligned_vertical_rules,
)
from .table_rules import (
    _count_occupied_closed_grid_columns as _count_occupied_closed_grid_columns,
)
from .table_rules import (
    _count_repeated_fill_bands as _count_repeated_fill_bands,
)
from .table_rules import (
    _count_stable_columns as _count_stable_columns,
)
from .table_rules import (
    _every_rule_interval_has_multi_cell_row as _every_rule_interval_has_multi_cell_row,
)
from .table_rules import (
    _expand_candidates_to_connected_rule_grids as _expand_candidates_to_connected_rule_grids,
)
from .table_rules import (
    _expand_rule_table_candidate as _expand_rule_table_candidate,
)
from .table_rules import (
    _full_height_vertical_rule_positions as _full_height_vertical_rule_positions,
)
from .table_rules import (
    _group_long_horizontal_rules as _group_long_horizontal_rules,
)
from .table_rules import (
    _longest_dense_multi_cell_rows as _longest_dense_multi_cell_rows,
)
from .table_rules import (
    _looks_like_page_column_prose as _looks_like_page_column_prose,
)
from .table_rules import (
    _median_fragment_height as _median_fragment_height,
)
from .table_rules import (
    _merge_table_candidates as _merge_table_candidates,
)
from .table_rules import (
    _rows_inside_rule_interval as _rows_inside_rule_interval,
)
from .table_rules import (
    _rule_bands_share_grid_tracks as _rule_bands_share_grid_tracks,
)
from .table_rules import (
    _rule_intervals_are_column_compatible as _rule_intervals_are_column_compatible,
)
from .table_rules import (
    _table_rows_align_with_rule_span as _table_rows_align_with_rule_span,
)
from .table_rules import (
    _table_segment_reaches_boundaries as _table_segment_reaches_boundaries,
)
from .table_text_styles import (
    render_native_table_html_with_scripts as render_native_table_html_with_scripts,
)
