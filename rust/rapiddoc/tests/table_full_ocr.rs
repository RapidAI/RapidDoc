use rapiddoc::legacy_model::{ocr_results, table_text};
use serde_json::{json, Value};
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/table-full-ocr.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_full_ocr_half_score_boundaries_and_normalization() {
    for c in fixture()["applies"].as_array().unwrap() {
        assert_eq!(
            table_text::apply_full_ocr(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_full_ocr_sort_cut_and_drop_policy() {
    for c in fixture()["cases"].as_array().unwrap() {
        let boxes =
            ocr_results::table_boxes(&json!({"boxes":c["boxes"],"masks":c["masks"]})).unwrap();
        if c["raises"] == true {
            assert!(boxes.as_array().unwrap().is_empty());
            continue;
        }
        let rows = boxes
            .as_array()
            .unwrap()
            .iter()
            .map(|_| json!([c["text"], c["score"]]))
            .collect::<Vec<_>>();
        assert_eq!(
            table_text::apply_full_ocr(&json!({"boxes":boxes,"results":rows})).unwrap(),
            c["expected"]
        );
    }
}
#[test]
fn invalid_full_ocr_results_fail_without_mutation() {
    let mut req = fixture()["applies"][0]["request"].clone();
    req["results"] = json!([]);
    let before = req.clone();
    assert!(table_text::apply_full_ocr(&req).is_err());
    assert_eq!(req, before);
    req["results"] = json!([["bad", "score"]]);
    assert!(table_text::apply_full_ocr(&req).is_err());
}
