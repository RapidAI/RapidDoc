use serde_json::{json, Value};
#[test]
fn actual_pdf_paths_to_graphic_candidates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-path-graphics.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
        let g = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &data, page, None, &limits).unwrap(),
        )
        .unwrap();
        assert_eq!(g["path_infos"], c["path_infos"], "paths {}", c["label"]);
        assert_eq!(g["page_size"], c["page_size"]);
        let out = rapiddoc::legacy_model::path_graphics::detect(
            &json!({"path_infos":g["path_infos"],"page_size":g["page_size"],"median_height":10.}),
        )
        .unwrap();
        assert_eq!(out, c["expected"], "candidates {}", c["label"]);
    }
}
