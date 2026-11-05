use serde_json::{json, Value};
#[test]
fn real_pdf_characters_to_script_payload_and_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-script-payloads.json")).unwrap(),
    )
    .unwrap();
    for c in f["real"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let geometry = rapiddoc_pdf::character_primitives::read_geometry(
            &library,
            &bytes,
            c["page_index"].as_u64().unwrap() as usize,
            None,
            true,
            &rapiddoc_core::Limits::default(),
        )
        .unwrap();
        let g = serde_json::to_value(geometry).unwrap();
        let lines=rapiddoc::legacy_model::native_lines::build(&json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"],"supported_angles":[0.,90.,180.,270.]})).unwrap();
        let prepared = rapiddoc::legacy_model::source_lines::prepare(
            &json!({"lines":lines,"page_size":g["page_size"],"drawing_lines":[],"image_infos":[]}),
        )
        .unwrap();
        let scripts=rapiddoc::legacy_model::inline_output::prepare_scripts(&json!({"lines":prepared["lines"],"page_size":g["page_size"],"tight_bboxes":g["tight_bboxes"],"origins":g["origins"],"all_chars":g["chars"],"drawing_lines":[]})).unwrap();
        assert_eq!(scripts, c["expected"], "payload angle {}", g["rotation"]);
        let blocks: Vec<_> = prepared["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| json!({"type":"text","content":l["text"],"bbox":l["bbox"]}))
            .collect();
        let out=rapiddoc::legacy_model::inline_output::trace(&json!({"blocks":blocks,"page_size":g["page_size"],"script_lines":scripts,"style_lines":[],"link_lines":[]})).unwrap();
        assert_eq!(out, c["output_expected"], "output angle {}", g["rotation"]);
    }
}
