use rapiddoc::legacy_model::closed_rule_pipeline as pipeline;
use serde_json::{json, Value};
#[test]
fn live_pdf_to_expanded_structural_tables() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-structural-tables.json")).unwrap(),
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
        let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start";
        let projected: Vec<Value> = lines
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                Value::Object(
                    fields
                        .split_whitespace()
                        .map(|k| (k.to_string(), l[k].clone()))
                        .collect(),
                )
            })
            .collect();
        let source = json!({"page_size":g["page_size"],"lines":projected,"chars":g["chars"],"drawing_lines":v["drawing_lines"],"path_infos":v["path_infos"],"form_bboxes":[],"image_bboxes":[],"signature_bboxes":[],"page_index":page});
        let mut q = c["request"].clone();
        assert_eq!(source, q["source"], "source {}", c["label"]);
        q["source"] = source;
        if !c["kind"].is_null() {
            let mut individual = q.clone();
            individual["kind"] = c["kind"].clone();
            assert_eq!(
                pipeline::detect_supplemental(&individual).unwrap(),
                c["expected"]["individual"],
                "individual {}",
                c["label"]
            );
        }
        for (key, call) in [(
            "page",
            pipeline::detect_structural_page as fn(&Value) -> Result<Value, rapiddoc_core::Error>,
        )] {
            assert_eq!(
                call(&q).unwrap(),
                c["expected"][key],
                "{key} {}",
                c["label"]
            );
        }
    }
}
