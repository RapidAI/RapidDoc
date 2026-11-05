use serde_json::{json, Value};
#[test]
fn actual_pdf_sources_to_raster_chart_groups() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-raster-charts.json")).unwrap(),
    )
    .unwrap();
    let mut facts = Vec::new();
    let mut image_pages = Vec::new();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
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
        assert_eq!(images["image_infos"], c["image_infos"]);
        let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let prepared=rapiddoc::legacy_model::source_lines::prepare(&json!({"lines":lines,"page_size":g["page_size"],"drawing_lines":[],"image_infos":images["image_infos"]})).unwrap();
        assert_eq!(
            prepared["lines"], c["prepared_lines"],
            "prepared {}",
            c["label"]
        );
        let lines:Vec<_>=prepared["lines"].as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"angle":l["angle"],"effective_height":l["effective_height"],"dominant_font_weight":l["dominant_font_weight"]})).collect();
        image_pages.push(
            json!({"lines":lines,"page_size":g["page_size"],"image_infos":images["image_infos"]}),
        );
        facts.push((g, prepared, data, page));
    }
    let filtered =
        rapiddoc::legacy_model::image_regions::prepare(&json!({"pages":image_pages})).unwrap();
    assert_eq!(
        filtered["watermark_fingerprints"],
        f["watermark_fingerprints"]
    );
    for (i, (g, p, data, page)) in facts.into_iter().enumerate() {
        let source = json!({"lines":p["lines"],"chars":g["chars"],"image_bboxes":filtered["image_bboxes"][i],"page_index":page});
        let evidence =
            rapiddoc::legacy_model::text_noise::prepare(&json!({"source":source})).unwrap();
        let paint = if evidence["candidates"].as_array().unwrap().is_empty() {
            Value::Null
        } else {
            let indices: Vec<i64> =
                serde_json::from_value(evidence["requested_char_indices"].clone()).unwrap();
            serde_json::to_value(
                rapiddoc_pdf::text_paint::read(&library, &data, page, &indices, None, &limits)
                    .unwrap(),
            )
            .unwrap()
        };
        let cleaned =
            rapiddoc::legacy_model::text_noise::apply(&json!({"source":source,"paint":paint}))
                .unwrap();
        let c = &f["cases"][i];
        assert_eq!(cleaned["source"], c["source"], "source {}", c["label"]);
        let out =
            rapiddoc::legacy_model::raster_charts::group(&json!({"source":cleaned["source"]}))
                .unwrap();
        assert_eq!(out, c["expected"], "charts {}", c["label"]);
    }
}
