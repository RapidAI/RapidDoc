use serde_json::{json, Value};
#[test]
fn actual_pdf_text_and_rules_to_prose_conflict_queries() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-prose-rule-conflicts.json")).unwrap(),
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
        let fields="text bbox source_index angle caption_start formula_candidate_only semantic_type dominant_font_weight";
        let lines: Vec<Value> = rows
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
        let ids: Vec<_> = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["source_index"].clone())
            .collect();
        let boundaries: Vec<_> = v["drawing_lines"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["orientation"] == "horizontal")
            .map(|r| r["bbox"].clone())
            .collect();
        let caption = if c["label"].as_str().unwrap().ends_with("caption-draft") {
            json!({})
        } else {
            Value::Null
        };
        let q = json!({"source":{"lines":lines,"drawing_lines":v["drawing_lines"]},"candidates":[{"kind":"draft","angle":0,"median_height":10.,"caption":caption,"boundaries":boundaries,"member_source_indices":ids},{"kind":"table","angle":0,"bbox":v["path_infos"][0]["bbox"],"core_bbox":null}]});
        assert_eq!(q, c["request"], "query {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::prose_rule_conflicts::exclude(&q).unwrap(),
            c["expected"],
            "filter {}",
            c["label"]
        );
    }
}
