use serde_json::{json, Value};
#[test]
fn actual_pdf_paths_to_filled_row_grid_candidates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-filled-grid.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &data, page, None, &limits).unwrap(),
        )
        .unwrap();
        let q =
            json!({"path_infos":v["path_infos"],"page_size":v["page_size"],"excluded_bboxes":[]});
        assert_eq!(q, c["request"], "source {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::filled_grid::detect(&q).unwrap(),
            c["expected"],
            "output {}",
            c["label"]
        );
    }
}
