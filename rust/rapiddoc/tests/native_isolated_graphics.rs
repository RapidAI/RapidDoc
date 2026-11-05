use serde_json::{json, Value};
#[test]
fn actual_pdf_sources_to_isolated_graphics() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-isolated-graphics.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &data, page, None, &limits).unwrap(),
        )
        .unwrap();
        let im = serde_json::to_value(
            rapiddoc_pdf::image_infos::read(&library, &data, page, None, &limits).unwrap(),
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
        let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"source_index":l["source_index"],"angle":l["angle"],"effective_height":l["effective_height"],"em_height":l["em_height"],"style_scale_repaired":l["style_scale_repaired"]})).collect();
        let boxes: Vec<_> = im["image_infos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["bbox"].clone())
            .collect();
        let source = json!({"lines":lines,"page_size":g["page_size"],"path_infos":v["path_infos"],"image_bboxes":boxes});
        assert_eq!(source, c["source"], "source {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::isolated_graphics::detect(&json!({"source":source})).unwrap(),
            c["expected"],
            "output {}",
            c["label"]
        );
    }
}
