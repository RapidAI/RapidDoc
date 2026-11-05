fn register_migration(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(office::parse_docx_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(office::parse_office_bytes, module)?)?;
    module.add_class::<pipeline::PdfEngine>()?;
    module.add_function(wrap_pyfunction!(render_pdf_model_page_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(classify_pdf_font_name_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(pdfium_font_runtime_info_json, module)?)?;
    module.add_function(wrap_pyfunction!(split_paragraphs_json, module)?)?;
    module.add_function(wrap_pyfunction!(sort_model_blocks_json, module)?)?;
    module.add_function(wrap_pyfunction!(legacy_host_pipeline_json, module)?)?;
    module.add_function(wrap_pyfunction!(native_classification_page_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        native_classification_difference_bytes,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(assemble_model_page_assets_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(fill_native_text_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(read_native_pdf_page_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_pdf_vector_geometry_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_pdf_form_geometry_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_pdf_link_annotations_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_pdf_image_infos_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_pdf_text_paint_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        deduplicate_native_characters_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        read_pdf_native_text_geometry_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        read_pdf_character_primitives_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(extract_original_pdf_images_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_model_crops_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_png, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_model_detection_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_model_recognition_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(replace_inline_image_assets, module)?)?;
    module.add_function(wrap_pyfunction!(
        materialize_native_table_media_page_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(native_glyph_weight_json, module)?)?;
    module.add_function(wrap_pyfunction!(native_pdf_document_json, module)?)?;
    module.add_function(wrap_pyfunction!(native_pdf_table_page_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        read_pdf_visible_text_geometry_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        materialize_native_table_media_pdf_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(organize_page_text_json, module)?)?;
    module.add_function(wrap_pyfunction!(select_table_candidate_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_classifier_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(decode_table_classification_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_recognition_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_formula_input_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_detection_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_direction_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(extract_table_native_text_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_model_payload_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(pad_table_detection_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_recognition_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(fix_unicode_text, module)?)?;
    module.add_class::<FormulaDecoder>()?;
    module.add_function(wrap_pyfunction!(
        prepare_recognition_batch_bgr_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prepare_detection_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(detection_boxes_json, module)?)?;
    module.add_function(wrap_pyfunction!(seal_geometry_json, module)?)?;
    module.add_function(wrap_pyfunction!(crop_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(parse_pdf_ocr_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_orientation_rgb_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_layout_rgb_json, module)?)?;
    module.add_function(wrap_pyfunction!(layout_rectangles_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_table_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(decode_table_structure_json, module)?)?;
    module.add_function(wrap_pyfunction!(render_wired_table_json, module)?)?;
    module.add_function(wrap_pyfunction!(blank_cell_crop_json, module)?)?;
    module.add_function(wrap_pyfunction!(match_cells_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_unet_bgr_json, module)?)?;
    module.add_function(wrap_pyfunction!(recover_table_polygons_json, module)?)?;
    module.add_function(wrap_pyfunction!(connect_table_lines_json, module)?)?;
    module.add_function(wrap_pyfunction!(table_components_json, module)?)?;
    module.add_function(wrap_pyfunction!(unet_line_geometry_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_unet_line_masks_json, module)?)?;
    module.add_function(wrap_pyfunction!(draw_table_lines_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(rotate_table_image_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(table_rotation_angle, module)?)?;
    module.add_function(wrap_pyfunction!(table_cells_from_lines_json, module)?)?;
    module.add_class::<PageImage>()?;
    module.add_class::<NativeTextPage>()?;
    module.add_class::<NativeTablePage>()?;
    Ok(())
}
