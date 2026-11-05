use serde_json::{json, Value};
fn fixture() -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/model-ocr-execution.json");
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
#[test]
fn old_detection_box_sort_merge_and_masks() {
    let f = fixture();
    assert_eq!(f["cases"].as_array().unwrap().len(), 131);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::ocr_results::boxes(&c["request"]).unwrap(),
            c["expected"],
            "boxes {i}"
        );
    }
}
#[test]
fn old_detection_attachment_and_recognition_policy() {
    let f = fixture();
    for (i, c) in f["preparations"].as_array().unwrap().iter().enumerate() {
        let p = rapiddoc::legacy_model::ocr_results::plan(&c["request"]).unwrap();
        assert_eq!(
            json!({"model":p["model"],"processed":p["processed"],"recognition":p["recognition"]}),
            c["expected"],
            "plan {i}"
        );
    }
    for (i, c) in f["applies"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::ocr_results::apply(&c["request"]).unwrap(),
            c["expected"],
            "policy {i}"
        );
    }
}
