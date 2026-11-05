use serde_json::{json, Value};
#[test]
fn actual_pdf_lines_and_paths_to_components() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-drawing-graphics.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let g = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(
                &library,
                &data,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                &rapiddoc_core::Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(g["drawing_lines"], c["drawing_lines"]);
        assert_eq!(g["path_infos"], c["path_infos"]);
        let out=rapiddoc::legacy_model::drawing_graphics::detect(&json!({"drawing_lines":g["drawing_lines"],"path_infos":g["path_infos"],"page_size":g["page_size"],"median_height":10.})).unwrap();
        assert_eq!(out, c["expected"], "{}", c["label"]);
    }
}
