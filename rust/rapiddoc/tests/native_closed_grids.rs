use serde_json::{json, Value};
#[test]
fn actual_pdf_text_and_rules_to_explicit_closed_grid_queries() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-closed-grids.json")).unwrap(),
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
        let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"bbox":l["bbox"],"source_index":l["source_index"],"effective_height":l["effective_height"]})).collect();
        let ids: Vec<_> = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["source_index"].clone())
            .collect();
        let p = &v["path_infos"][0];
        let frame = p["rectangle_bboxes"]
            .as_array()
            .unwrap()
            .first()
            .unwrap_or(&p["bbox"]);
        let q = json!({"source":{"form_bboxes":[],"drawing_lines":v["drawing_lines"],"lines":lines},"candidates":[{"bbox":frame,"core_bbox":null,"angle":0,"line_indices":ids}],"allow_two_by_two":c["request"]["allow_two_by_two"]});
        assert_eq!(q, c["request"], "live query {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::closed_grid::inspect(&q).unwrap(),
            c["expected"],
            "proof {}",
            c["label"]
        );
    }
}
