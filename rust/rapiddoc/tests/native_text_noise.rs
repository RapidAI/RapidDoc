use serde_json::{json, Value};
#[test]
fn actual_pdf_paint_and_source_noise() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-text-noise.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let page = c["page_index"].as_u64().unwrap() as usize;
        let indices: Vec<i64> = serde_json::from_value(c["indices"].clone()).unwrap();
        let paint = rapiddoc_pdf::text_paint::read(
            &library,
            &bytes,
            page,
            &indices,
            None,
            &rapiddoc_core::Limits::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(paint).unwrap(),
            c["paint"],
            "paint {}",
            c["label"]
        );
        let g = serde_json::to_value(
            rapiddoc_pdf::character_primitives::read_geometry(
                &library,
                &bytes,
                page,
                None,
                true,
                &rapiddoc_core::Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let rows = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let prepared = rapiddoc::legacy_model::source_lines::prepare(
            &json!({"lines":rows,"page_size":g["page_size"],"drawing_lines":[],"image_infos":[]}),
        )
        .unwrap();
        let source = json!({"lines":prepared["lines"],"chars":g["chars"],"page_index":page});
        assert_eq!(source, c["source"], "source {}", c["label"]);
        let evidence =
            rapiddoc::legacy_model::text_noise::prepare(&json!({"source":source})).unwrap();
        assert_eq!(evidence, c["prepared"], "candidates {}", c["label"]);
        let requested: Vec<i64> =
            serde_json::from_value(evidence["requested_char_indices"].clone()).unwrap();
        let paint = rapiddoc_pdf::text_paint::read(
            &library,
            &bytes,
            page,
            &requested,
            None,
            &rapiddoc_core::Limits::default(),
        )
        .unwrap();
        let output =
            rapiddoc::legacy_model::text_noise::apply(&json!({"source":source,"paint":paint}))
                .unwrap();
        assert_eq!(output, c["expected"], "noise {}", c["label"]);
    }
}
