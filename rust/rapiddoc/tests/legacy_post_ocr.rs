use rapiddoc::legacy_post_ocr::{apply, plan};
use serde_json::{json, Value};
#[test]
fn actual_old_post_ocr_selection_and_fp64_rounding_match() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-post-ocr.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let images: Vec<_> = plan(&c["pages"])
            .unwrap()
            .into_iter()
            .map(|t| t.image)
            .collect();
        assert_eq!(json!(images), c["targets"], "{}", c["name"]);
        assert_eq!(
            apply(&json!({"pages":c["pages"],"results":c["results"]})).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}
#[test]
fn result_shape_and_budgets_are_checked() {
    let pages = json!([{"preproc_blocks":[{"type":"text","lines":[{"spans":[{"np_img":"id"}]}]}],"discarded_blocks":[]}]);
    for results in [
        json!([]),
        json!([["x"]]),
        json!([["x", "bad"]]),
        json!([[12, 0.9]]),
    ] {
        assert!(apply(&json!({"pages":pages,"results":results})).is_err());
    }
    assert!(plan(&json!({})).is_err());
    assert!(plan(&Value::Array(vec![
        json!({"preproc_blocks":[],"discarded_blocks":[]});
        1001
    ]))
    .is_err());
}
