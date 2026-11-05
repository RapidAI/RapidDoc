use serde_json::{json, Value};
#[test]
fn actual_pdf_chars_paths_lines_to_tiled_tables_and_bar_exclusion() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-tiled-tables.json")).unwrap(),
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
        let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"source_index":l["source_index"],"chars":l["chars"]})).collect();
        let source = json!({"page_size":g["page_size"],"chars":g["chars"],"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"lines":lines});
        assert_eq!(source, c["request"]["source"], "source {}", c["label"]);
        // Public entrypoints validate independent callers; installed tests also chain the APIs.
        let out = rapiddoc::legacy_model::tiled_tables::detect(
            &json!({"source":source,"excluded_bboxes":[]}),
        )
        .unwrap();
        assert_eq!(out, c["expected"], "output {}", c["label"]);
        let boxes: Vec<_> = out["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["bbox"].clone())
            .collect();
        if !boxes.is_empty() {
            let before=rapiddoc::legacy_model::bar_geometry::prepare(&json!({"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"em":10.,"table_bboxes":[]})).unwrap();
            assert!(!before["admitted_bars"].as_array().unwrap().is_empty());
            let after=rapiddoc::legacy_model::bar_geometry::prepare(&json!({"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"em":10.,"table_bboxes":boxes})).unwrap();
            assert!(after["admitted_bars"].as_array().unwrap().is_empty());
        }
    }
}
