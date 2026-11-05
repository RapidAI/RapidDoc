use serde_json::{json, Value};
#[test]
fn live_pdf_rows_and_rules_to_annotation_preparation() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-rule-annotation-preparation.json")).unwrap(),
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
        let frame = &v["path_infos"][0]["bbox"];
        let mut rows = Vec::new();
        let mut core = Vec::new();
        let mut caption = Value::Null;
        let mut foot = Vec::new();
        for (i, l) in lines.as_array().unwrap().iter().enumerate() {
            let b = &l["bbox"];
            let cy = (b[1].as_f64().unwrap() + b[3].as_f64().unwrap()) / 2.;
            let cx = (b[0].as_f64().unwrap() + b[2].as_f64().unwrap()) / 2.;
            if frame[0].as_f64().unwrap() <= cx
                && cx <= frame[2].as_f64().unwrap()
                && frame[1].as_f64().unwrap() <= cy
                && cy <= frame[3].as_f64().unwrap()
            {
                core.push(i);
            }
            let text = l["text"].as_str().unwrap();
            if caption.is_null() && text.starts_with("Table") {
                caption = l["source_index"].clone();
            }
            if text.starts_with("Note") {
                foot.push(i);
            }
            rows.push(json!({"fragments":[{"text":l["text"],"bbox":b,"local_bbox":b,"line_index":l["source_index"],"visual_row_id":null}],"center_y":cy,"bbox":b,"visual_row_id":l["visual_row_id"]}));
        }
        let rules: Vec<_> = v["drawing_lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["bbox"].clone())
            .collect();
        let q = json!({"rows":rows,"rule_bboxes":rules,"core_row_indices":core,"median_height":10.,"caption_source_index":caption,"footnote_row_indices":foot});
        assert_eq!(q, c["request"], "query {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::rule_annotation_preparation::prepare(&q).unwrap(),
            c["expected"],
            "annotations {}",
            c["label"]
        );
    }
}
