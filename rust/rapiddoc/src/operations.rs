//! Shared JSON operation dispatch; language bindings only serialize arguments.
use rapiddoc_core::Error;
use serde_json::Value;

pub fn run(operation: &str, value: Value) -> Result<Value, Error> {
    let result = match operation {
        "guess_code_language" => {
            let texts = value["texts"]
                .as_array()
                .ok_or_else(crate::legacy_page::invalid)?;
            let runtime = value["runtime"].as_str().map(std::path::Path::new);
            let labels = texts
                .iter()
                .map(|text| {
                    let text = text.as_str().ok_or_else(crate::legacy_page::invalid)?;
                    rapiddoc_inference::magika::language(text, runtime)
                })
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::json!(labels)
        }
        "assemble_page" => crate::legacy_page::assemble(&value)?,
        "convert_layout" => crate::legacy_model::convert(&value)?,
        "filter_layout" => crate::legacy_model::filter(&value)?,
        "native_ocr_boxes" => crate::legacy_model::native_ocr::boxes(&value)?,
        "plan_native_ocr" => crate::legacy_model::native_ocr::plan(&value)?,
        "model_detection_boxes" => crate::legacy_model::ocr_results::boxes(&value)?,
        "table_detection_boxes" => crate::legacy_model::ocr_results::table_boxes(&value)?,
        "plan_table_direction" => crate::legacy_model::table_direction::plan(&value)?,
        "select_table_orientation" => {
            crate::legacy_model::table_direction::select_model_rotation(&value)?
        }
        "table_native_spans" => crate::legacy_model::table_native::spans(&value)?,
        "plan_table_hints" => crate::legacy_model::table_hints::plan(&value)?,
        "legacy_table_match" => crate::legacy_model::table_match::run(&value)?,
        "select_legacy_table" => crate::legacy_model::table_selection::select(&value)?,
        "plan_native_table_rule" => crate::legacy_model::table_native_rule::plan(&value)?,
        "native_table_pending" => crate::legacy_model::table_native_text::pending(&value)?,
        "build_native_table_text" => crate::legacy_model::table_native_text::build(&value)?,
        "prepare_native_table_rule_evidence" => {
            crate::legacy_model::table_native_vector::prepare(&value)?
        }
        "canonicalize_native_table_tracks" => {
            crate::legacy_model::table_native_vector::canonicalize(&value)?
        }
        "build_native_table_vector_candidate" => {
            crate::legacy_model::table_native_vector::build(&value)?
        }
        "generate_native_table_vector_candidates" => {
            crate::legacy_model::table_native_vector::generate(&value)?
        }
        "generate_native_table_text_candidates" => {
            crate::legacy_model::table_native_text_grid::generate(&value)?
        }
        "regroup_native_table_numeric_text" => {
            crate::legacy_model::table_native_aligned_numeric::regroup(&value)?
        }
        "build_native_table_aligned_numeric_candidate" => {
            crate::legacy_model::table_native_aligned_numeric::generate(&value)?
        }
        "generate_native_table_rule_band_candidates" => {
            crate::legacy_model::table_native_rule_band::generate(&value)?
        }
        "generate_native_table_sparse_hybrid_candidates" => {
            crate::legacy_model::table_native_sparse_hybrid::generate(&value)?
        }
        "generate_native_table_multiline_candidates" => {
            crate::legacy_model::table_native_multiline::generate(&value)?
        }
        "build_native_table_banded_numeric_candidate" => {
            crate::legacy_model::table_native_banded_numeric::generate(&value)?
        }
        "project_native_table_spatial_text" => {
            crate::legacy_model::table_native_multiline::project_text(&value)?
        }
        "inspect_native_table_sparse_columns" => {
            crate::legacy_model::table_native_rule_band::inspect_columns(&value)?
        }
        "inspect_native_table_engine_policy" => {
            crate::legacy_model::table_native_engine_policy::inspect(&value)?
        }
        "evaluate_native_table_primitives" => {
            crate::legacy_model::table_native_engine::evaluate_json(&value)?
        }
        "recover_native_table_primitives" => {
            crate::legacy_model::table_native_engine::recover(&value)?
        }
        "diagnose_native_table_primitives" => {
            crate::legacy_model::table_native_engine::inspect(&value)?
        }
        "inspect_native_table_vector_tracks" => {
            crate::legacy_model::table_native_vector::inspect_tracks(&value)?
        }
        "inspect_native_table_candidate" => {
            crate::legacy_model::table_native_candidate::inspect(&value)?
        }
        "build_native_table_candidate" => {
            crate::legacy_model::table_native_candidate::build(&value)?
        }
        "render_native_table_candidate" => {
            serde_json::Value::String(crate::legacy_model::table_native_candidate::render(&value)?)
        }
        "merge_native_inline_clusters" => {
            crate::legacy_model::table_text_styles::clusters::merge(&value)?
        }
        "prepare_native_table_block" => crate::legacy_model::table_block::prepare(&value)?,
        "materialize_native_table_blocks" => {
            crate::legacy_model::table_block::commit::materialize(&value)?
        }
        "build_native_visual_lines" => crate::legacy_model::native_lines::build(&value)?,
        "prepare_native_source_lines" => crate::legacy_model::source_lines::prepare(&value)?,
        "prepare_native_text_noise" => crate::legacy_model::text_noise::prepare(&value)?,
        "apply_native_text_noise" => crate::legacy_model::text_noise::apply(&value)?,
        "detect_native_path_graphics" => crate::legacy_model::path_graphics::detect(&value)?,
        "inspect_native_rule_table_geometry" => {
            crate::legacy_model::closed_rule_pipeline::inspect(&value)?
        }
        "expand_native_connected_table_grids" => {
            crate::legacy_model::closed_rule_pipeline::expand(&value)?
        }
        "detect_native_closed_rule_tables" => {
            crate::legacy_model::closed_rule_pipeline::detect(&value)?
        }
        "detect_native_physical_table_candidates" => {
            crate::legacy_model::closed_rule_pipeline::detect_page(&value)?
        }
        "detect_native_rule_table_candidates" => {
            crate::legacy_model::closed_rule_pipeline::detect_rules(&value)?
        }
        "detect_native_supplemental_tables" => {
            crate::legacy_model::closed_rule_pipeline::detect_supplemental(&value)?
        }
        "detect_native_structural_table_candidates" => {
            crate::legacy_model::closed_rule_pipeline::detect_structural_page(&value)?
        }
        "detect_native_table_candidates" => {
            crate::legacy_model::closed_rule_pipeline::detect_complete_page(&value)?
        }
        "apply_native_table_page_policies" => {
            crate::legacy_model::closed_rule_pipeline::apply_page_policies(&value)?
        }
        "trailing_native_table_grid" => {
            crate::legacy_model::closed_rule_pipeline::trailing_table_grid(&value)?
        }
        "expand_native_rule_table_candidate" => {
            crate::legacy_model::rule_candidate_expansion::expand(&value)?
        }
        "collect_native_table_footnotes" => crate::legacy_model::table_footnotes::collect(&value)?,
        "prepare_native_rule_table_annotations" => {
            crate::legacy_model::rule_annotation_preparation::prepare(&value)?
        }
        "exclude_native_prose_spanning_rule_drafts" => {
            crate::legacy_model::prose_rule_conflicts::exclude(&value)?
        }
        "merge_native_table_candidates" => {
            crate::legacy_model::table_candidate_merge::merge(&value)?
        }
        "prefer_complete_native_table_replacements" => {
            crate::legacy_model::table_replacements::prefer(&value)?
        }
        "inspect_native_closed_grids" => crate::legacy_model::closed_grid::inspect(&value)?,
        "detect_native_filled_grid_tables" => crate::legacy_model::filled_grid::detect(&value)?,
        "protect_native_graphic_captions" => {
            crate::legacy_model::graphic_captions::prepare(&value)?
        }
        "detect_native_bar_graphics" => crate::legacy_model::bar_graphics::detect(&value)?,
        "prepare_native_bar_groups" => crate::legacy_model::bar_groups::prepare(&value)?,
        "detect_native_tiled_tables" => crate::legacy_model::tiled_tables::detect(&value)?,
        "prepare_native_bar_geometry" => crate::legacy_model::bar_geometry::prepare(&value)?,
        "detect_native_diagram_components" => {
            crate::legacy_model::diagram_components::detect(&value)?
        }
        "detect_native_strong_graphics" => crate::legacy_model::strong_graphics::detect(&value)?,
        "prepare_native_table_page" => crate::legacy_model::table_page::prepare(&value)?,
        "build_native_code_blocks" => crate::legacy_model::code_containers::build(&value)?,
        "prepare_native_code_table_page" => {
            crate::legacy_model::table_page::prepare_code_tables(&value)?
        }
        "prepare_native_document_containers" => {
            crate::legacy_model::page_containers::prepare_document(&value)?
        }
        "build_native_document_geometry_plan" => {
            crate::legacy_model::native_geometry::build_document_plan(&value)?
        }
        "prepare_native_document_geometry" => {
            crate::legacy_model::native_geometry::prepare_document(&value)?
        }
        "prepare_native_page_marginals" => crate::legacy_model::native_marginals::prepare(&value)?,
        "prepare_native_document_sources" => {
            crate::legacy_model::native_marginals::prepare_document_sources(&value)?
        }
        "prepare_native_document_pages" => {
            crate::legacy_model::native_document::prepare_pages(&value)?
        }
        "classify_native_document_auxiliary" => {
            crate::legacy_model::native_document_text::prepare(&value)?
        }
        "prepare_native_document_auxiliary" => {
            crate::legacy_model::native_document::prepare_auxiliary(&value)?
        }
        "prepare_native_document_text" => {
            crate::legacy_model::native_document::prepare_text(&value)?
        }
        "prepare_native_document_references" => {
            crate::legacy_model::native_document::prepare_references(&value)?
        }
        "classify_native_document_references" => {
            crate::legacy_model::native_references::prepare(&value)?
        }
        "group_native_reference_lines" => crate::legacy_model::native_references::group(&value)?,
        "recover_native_formula_blocks" => {
            crate::legacy_model::native_formula_components::build_full(&value)?
        }
        "prepare_native_formula_input" => {
            crate::legacy_model::native_formula_components::prepare_input(&value)?
        }
        "prepare_native_final_page_lines" => {
            crate::legacy_model::native_finalization::prepare_lines(&value)?
        }
        "prepare_native_final_document_lines" => {
            crate::legacy_model::native_document::prepare_final_text(&value)?
        }
        "build_native_text_blocks" => {
            crate::legacy_model::native_text_assembly::build_text_blocks(&value)?
        }
        "prepare_native_final_page_blocks" => {
            crate::legacy_model::native_finalization::prepare_blocks(&value)?
        }
        "prepare_native_final_document_blocks" => {
            crate::legacy_model::native_document::prepare_text_blocks(&value)?
        }
        "refine_native_text_blocks" => {
            crate::legacy_model::native_text_assembly::refine_blocks(&value)?
        }
        "refine_native_page_blocks" => {
            crate::legacy_model::native_text_assembly::refine_page_blocks(&value)?
        }
        "native_final_output" => crate::legacy_model::native_final_output::run(&value)?,
        "native_visual_annotations" => crate::legacy_model::native_visual_annotations::run(&value)?,
        "bind_native_visual_annotations" => {
            crate::legacy_model::native_visual_annotations::bind(&value)?
        }
        "recover_native_annotation_bands" => {
            crate::legacy_model::native_visual_annotations::recover_bands(&value)?
        }
        "native_grouped_order" => crate::legacy_model::native_grouped_order::run(&value)?,
        "prepare_native_spacing_lines" => {
            crate::legacy_model::inline_output::prepare_spacing(&value)?
        }
        "apply_native_spacing_lines" => crate::legacy_model::inline_output::apply_spacing(&value)?,
        "materialize_native_document_inline" => {
            crate::legacy_model::native_inline_document::materialize(&value)?
        }
        "plan_native_glyph_weight" => crate::legacy_model::native_glyph_weight::plan(&value)?,
        "convert_native_pdf_layout" => crate::legacy_model::native_adapter::convert(&value)?,
        "native_pdf_layout_fallback" => crate::legacy_model::native_adapter::fallback(&value)?,
        "merge_native_pdf_supplements" => crate::legacy_model::native_adapter::supplements(&value)?,
        "native_pdf_table_text_angle" => crate::legacy_model::native_adapter::direction(&value)?,
        "recover_native_pdf_table_region" => {
            crate::legacy_model::native_adapter::recover_table_region_owned(value)?
        }
        "normalize_native_pdf_model_text" => {
            crate::legacy_model::native_model_normalization::normalize_pages(&value)?
        }
        "inspect_native_text_rows" => {
            crate::legacy_model::native_text_assembly::inspect_rows(&value)?
        }
        "classify_native_page_structure" => {
            crate::legacy_model::native_titles::classify_page_structure(&value)?
        }
        "refine_native_final_titles" => {
            crate::legacy_model::native_titles::refine_final_titles(&value)?
        }
        "merge_native_final_text" => {
            crate::legacy_model::native_page_preparation::merge_final_text(&value)?
        }
        "prepare_native_page_formulas" => {
            crate::legacy_model::native_formula_components::prepare_page(&value)?
        }
        "prepare_native_document_formulas" => {
            crate::legacy_model::native_document::prepare_formulas(&value)?
        }
        "classify_native_page_titles" => crate::legacy_model::native_titles::prepare(&value)?,
        "infer_native_document_title_profile" => {
            crate::legacy_model::native_titles::infer_document_profile(&value)?
        }
        "classify_native_document_titles" => {
            crate::legacy_model::native_titles::classify_document(&value)?
        }
        "apply_native_geometry_plan" => crate::legacy_model::native_geometry::apply(&value)?,
        "finish_native_geometry_page" => crate::legacy_model::native_geometry::finish_page(&value)?,
        "extract_native_index_blocks" => {
            crate::legacy_model::page_containers::extract_index(&value)?
        }
        "select_native_form_regions" => crate::legacy_model::page_containers::select_forms(&value)?,
        "prepare_native_container_table_page" => {
            crate::legacy_model::page_containers::prepare(&value)?
        }
        "detect_native_raster_diagrams" => crate::legacy_model::raster_diagrams::detect(&value)?,
        "detect_native_concentric_graphics" => {
            crate::legacy_model::concentric_graphics::detect(&value)?
        }
        "arbitrate_native_graphic_candidates" => {
            crate::legacy_model::graphic_arbitration::select(&value)?
        }
        "inspect_native_graphic_grids" => crate::legacy_model::graphic_grid::inspect(&value)?,
        "detect_native_isolated_graphics" => {
            crate::legacy_model::isolated_graphics::detect(&value)?
        }
        "detect_native_drawing_graphics" => crate::legacy_model::drawing_graphics::detect(&value)?,
        "detect_native_stacked_graphics" => crate::legacy_model::stacked_graphics::detect(&value)?,
        "group_native_raster_chart_descriptions" => {
            crate::legacy_model::raster_charts::group(&value)?
        }
        "prepare_native_image_regions" => crate::legacy_model::image_regions::prepare(&value)?,
        "detect_native_inline_evidence" => crate::legacy_model::inline_evidence::detect(&value)?,
        "apply_native_inline_evidence" => crate::legacy_model::inline_output::apply(&value)?,
        "trace_native_inline_evidence" => crate::legacy_model::inline_output::trace(&value)?,
        "prepare_native_script_lines" => {
            crate::legacy_model::inline_output::prepare_scripts(&value)?
        }
        "prepare_native_table_annotations" => {
            crate::legacy_model::table_block::prepare_annotations(&value)?
        }
        "restore_native_front_matter_panel" => {
            crate::legacy_model::table_block::output::restore(&value)?
        }
        "classify_native_table_script_cell" => {
            crate::legacy_model::table_text_styles::clusters::cell_roles(&value)?
        }
        "infer_native_text_lanes" => {
            crate::legacy_model::table_text_styles::lanes::infer_json(&value)?
        }
        "classify_native_table_script_line" => {
            crate::legacy_model::table_text_styles::preparation::classify_cell_line(&value)?
        }
        "measure_native_line_typography" => {
            crate::legacy_model::table_text_styles::preparation::measure(&value)?
        }
        "prepare_native_table_script_cells" => {
            crate::legacy_model::table_text_styles::preparation::prepare(&value)?
        }
        "refine_native_script_tokens" => {
            crate::legacy_model::table_text_styles::refine_tokens(&value)?
        }
        "classify_native_script_runs" => {
            crate::legacy_model::table_text_styles::classify_runs(&value)?
        }
        "classify_native_script_line" => {
            crate::legacy_model::table_text_styles::classify_line(&value)?
        }
        "native_fraction_member_sources" => {
            crate::legacy_model::table_text_styles::fraction_sources(&value)?
        }
        "classify_native_script_roles" => {
            crate::legacy_model::table_text_styles::classify_geometry(&value)?
        }
        "native_numeric_script_indices" => {
            crate::legacy_model::table_text_styles::numeric_indices(&value)?
        }
        "render_native_table_styled_cell" => {
            serde_json::Value::String(crate::legacy_model::table_text_styles::render_cell(&value)?)
        }
        "render_native_table_html_with_scripts" => serde_json::Value::String(
            crate::legacy_model::table_text_styles::materialization::render(&value)?,
        ),
        "recover_native_table_result" => {
            crate::legacy_model::table_text_styles::materialization::recover_owned(value)?
        }
        "build_native_table_cell_text" => {
            crate::legacy_model::table_native_text::cell_parts(&value)?
        }
        "accept_native_table_rule" => crate::legacy_model::table_native_rule::accept(&value)?,
        "select_native_table_candidates" => crate::legacy_model::table_native_rule::select(&value)?,
        "legacy_table_match_geometry" => crate::legacy_model::table_match::geometry(&value)?,
        "legacy_table_match_assemble" => crate::legacy_model::table_match::assemble(&value)?,
        "normalize_legacy_table_html" => match value.as_str() {
            Some(html) => {
                serde_json::Value::String(crate::legacy_model::table_normalize::html(html)?)
            }
            None if value.is_null() => serde_json::Value::Null,
            _ => {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: "表格 HTML 必须为字符串或 null".into(),
                })
            }
        },
        "restore_table_detection" => crate::legacy_model::table_text::restore(&value)?,
        "apply_table_ocr" => crate::legacy_model::table_text::apply_ocr(&value)?,
        "apply_table_full_ocr" => crate::legacy_model::table_text::apply_full_ocr(&value)?,
        "apply_table_prediction" => crate::legacy_model::table_text::apply_prediction(&value)?,
        "plan_model_recognition" => crate::legacy_model::ocr_results::plan(&value)?,
        "apply_model_recognition" => crate::legacy_model::ocr_results::apply(&value)?,
        "plan_model_formulas" => crate::legacy_model::formulas::plan(&value)?,
        "apply_model_formulas" => crate::legacy_model::formulas::apply(&value)?,
        "assemble_pages" => crate::legacy_page::assemble_many(&value)?,
        "assemble_pages_with_formulas" => crate::legacy_page::assemble_many_with_formulas(&value)?,
        "optimize_formulas" => crate::legacy_page::optimize_formula_numbers(&value)?,
        "render_pages" => crate::legacy_page::render(&value)?,
        "merge_tables" => crate::legacy_table_merge::merge(&value)?,
        "plan_post_ocr" => {
            serde_json::to_value(crate::legacy_post_ocr::plan(&value)?).map_err(|_| {
                rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: "OCR 计划序列化失败".into(),
                }
            })?
        }
        "apply_post_ocr" => crate::legacy_post_ocr::apply(&value)?,
        _ => {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::Unsupported,
                message: "未知旧页面操作".into(),
            })
        }
    };
    Ok(result)
}
