use rapiddoc::legacy_page::assets::{assemble, fill_native_text};
use serde_json::Value;
#[test]
fn actual_old_native_span_filling_and_complete_pages_match() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "legacy-native-text",
        "legacy-native-text-narrow",
        "legacy-native-text-metadata",
        "legacy-native-text-numeric",
    ] {
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{name}.json"))).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let output = fill_native_text(
                &c["request"],
                &std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap(),
            )
            .unwrap();
            assert_eq!(output.spans, c["expected"], "{name}/{}", c["name"]);
            assert_eq!(
                output.images.len(),
                c["crops"].as_array().unwrap().len(),
                "{}",
                c["name"]
            );
        }
        for c in f["pages"].as_array().unwrap() {
            let output = assemble(
                &c["request"],
                &std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap(),
                &[],
                false,
                0.9,
            )
            .unwrap();
            assert_eq!(output.page, c["expected"], "{}", c["name"]);
            assert_eq!(
                output.ocr_images.len(),
                c["crops"].as_array().unwrap().len()
            );
        }
    }
}

#[test]
fn extreme_text_height_and_invalid_facts_return_errors_before_crops() {
    use serde_json::json;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let png = std::fs::read(root.join("build/migration/native-text-images/high.png")).unwrap();
    let span = |bbox| json!({"type":"text","bbox":bbox,"content":"","score":0.9});
    let request = json!({"spans":[span([0.,0.,1.,1e-300]),span([0.,0.,1.,1e-300]),span([-1e6,-1e6,1e6,1e6])],"blocks":[[-1e6,-1e6,1e6,1e6,null,null,null,"text"]],"discarded_blocks":[],"native_page":{"blocks":[]},"scale":1.,"page_index":7});
    assert!(fill_native_text(&request, &png).is_err());
    let bad = json!({"spans":[null],"blocks":[],"discarded_blocks":[],"native_page":{"blocks":[]},"scale":1.,"page_index":7});
    assert!(fill_native_text(&bad, &png).is_err());
}
