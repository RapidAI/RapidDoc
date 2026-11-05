use serde_json::{json, Value};
#[test]
fn actual_pdf_full_bar_detection_and_caption_mutations() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-bar-graphics.json")).unwrap(),
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
        let images = serde_json::to_value(
            rapiddoc_pdf::image_infos::read(&library, &data, page, None, &limits).unwrap(),
        )
        .unwrap();
        let rows = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let fields="text bbox angle source_index chars font_signature dominant_font_weight effective_height em_height style_scale_repaired semantic_type title_suppressed paragraph_group visual_row_id";
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
        let boxes: Vec<_> = images["image_infos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|im| im["bbox"].clone())
            .collect();
        let source = json!({"page_size":g["page_size"],"chars":g["chars"],"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"lines":lines,"image_bboxes":boxes,"form_bboxes":[]});
        assert_eq!(source, c["request"]["source"], "source {}", c["label"]);
        let q = json!({"source":source,"em":10.});
        assert_eq!(
            rapiddoc::legacy_model::bar_graphics::detect(&q).unwrap(),
            c["expected"],
            "full bars {}",
            c["label"]
        );
        if c["grid_rejected"] == true {
            assert!(
                !rapiddoc::legacy_model::bar_groups::prepare(&q).unwrap()["groups"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
