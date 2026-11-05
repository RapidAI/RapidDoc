fn read_request(f: &Value) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bytes = std::fs::read(root.join(f["pdf"].as_str().unwrap())).unwrap();
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let mut pages = Vec::new();
    for (i, expected) in f["request"]["pages"].as_array().unwrap().iter().enumerate() {
        let g = serde_json::to_value(
            rapiddoc_pdf::character_primitives::read_geometry(
                &library,
                &bytes,
                i,
                None,
                true,
                &Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &bytes, i, None, &Limits::default())
                .unwrap(),
        )
        .unwrap();
        let forms = serde_json::to_value(
            rapiddoc_pdf::form_structure::read(&library, &bytes, i, None, &Limits::default())
                .unwrap(),
        )
        .unwrap();
        let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let source = json!({"page_size":g["page_size"],"page_index":i,"lines":lines,"chars":g["chars"],"drawing_lines":v["drawing_lines"],"path_infos":v["path_infos"],"form_infos":forms["form_infos"],"form_bboxes":forms["form_bboxes"],"image_bboxes":[],"signature_bboxes":[]});
        assert_eq!(source, expected["source"], "live source {i}");
        let mut geometry =
            json!({"chars":g["chars"],"tight_bboxes":[],"origins":[],"loose_bboxes":[]});
        for (key, value) in [
            ("tight_bboxes", "bbox"),
            ("origins", "origin"),
            ("loose_bboxes", "bbox"),
        ] {
            let mut rows = g[key]
                .as_object()
                .unwrap()
                .iter()
                .map(|(index, b)| {
                    let mut row = json!({"char_idx":index.parse::<usize>().unwrap()});
                    row[value] = b.clone();
                    row
                })
                .collect::<Vec<_>>();
            rows.sort_by_key(|r| r["char_idx"].as_u64().unwrap());
            geometry[key] = json!(rows);
        }
        assert_eq!(geometry, expected["geometry"], "live geometry {i}");
        pages.push(json!({"source":source,"geometry":geometry}));
    }
    json!({"pages":pages})
}
use rapiddoc_core::Limits;
use serde_json::{json, Value};
#[test]
fn live_pdfium_document_pages() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-document-pages-runtime.json")).unwrap(),
    )
    .unwrap();
    let mut request = read_request(&f);
    let actual = rapiddoc::legacy_model::native_document::prepare_pages(&request).unwrap();
    assert_eq!(actual, f["expected"]);
    let text: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-post-text-document.json")).unwrap(),
    )
    .unwrap();
    let case = text["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "live-marginal-pages")
        .unwrap();
    request["include_post_text_roles"] = json!(true);
    let actual = rapiddoc::legacy_model::native_document::prepare_pages(&request).unwrap();
    assert_eq!(actual, case["expected"]);
}

#[test]
fn live_pdfium_document_footnotes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-footnote-runtime.json")).unwrap(),
    )
    .unwrap();
    let mut request = read_request(&f);
    request["include_post_footnotes"] = json!(true);
    let actual = rapiddoc::legacy_model::native_document::prepare_pages(&request).unwrap();
    assert_eq!(actual, f["expected"]);
    assert_eq!(actual["pages"][0]["page_footnote_groups"], json!([[6, 7]]));
    assert_eq!(
        actual["pages"][0]["remaining_lines"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|l| l["semantic_type"] == "page_footnote")
            .count(),
        2
    );
}
#[test]
fn live_pdfium_complete_prepared_page() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-footnote-runtime.json")).unwrap(),
    )
    .unwrap();
    let mut request = read_request(&f);
    request["prepare_page"] = json!(true);
    let actual = rapiddoc::legacy_model::native_document::prepare_pages(&request).unwrap();
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-prepared-pages.json")).unwrap(),
    )
    .unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "live-footnote-page")
        .unwrap();
    assert_eq!(actual, case["expected"]);
    assert!(actual["pages"][0]["remaining_lines"]
        .as_array()
        .unwrap()
        .iter()
        .all(|l| l["chars"].as_array().unwrap().is_empty()));
    assert_eq!(actual["pages"][0]["page_footnote_groups"], json!([[6, 7]]));
    assert!(!actual["pages"][0]["script_lines"]
        .as_array()
        .unwrap()
        .is_empty());
}
