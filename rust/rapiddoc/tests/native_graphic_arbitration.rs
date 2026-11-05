use serde_json::{json, Value};
#[test]
fn actual_pdf_component_candidates_and_grid_arbitration() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-graphic-arbitration.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &data, page, None, &limits).unwrap(),
        )
        .unwrap();
        let g = serde_json::to_value(
            rapiddoc_pdf::character_primitives::read_geometry(
                &library, &data, page, None, true, &limits,
            )
            .unwrap(),
        )
        .unwrap();
        let rows = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"angle":l["angle"],"source_index":l["source_index"],"visual_row_id":l["visual_row_id"]})).collect();
        let source = json!({"page_size":g["page_size"],"lines":lines,"drawing_lines":v["drawing_lines"],"form_bboxes":[]});
        assert_eq!(source, c["source"], "source {}", c["label"]);
        let generated=rapiddoc::legacy_model::drawing_graphics::detect(&json!({"drawing_lines":v["drawing_lines"],"path_infos":v["path_infos"],"page_size":g["page_size"],"median_height":10.})).unwrap();
        assert_eq!(generated["graphic_bboxes"], c["candidate_bboxes"]);
        assert_eq!(rapiddoc::legacy_model::graphic_arbitration::select(&json!({"source":source,"candidate_bboxes":generated["graphic_bboxes"],"median_height":10.})).unwrap(),c["expected"],"arbitration {}",c["label"]);
    }
}
