use rapiddoc::legacy_model::table_match;
use serde_json::{json, Value};
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/legacy-table-match.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_iou_inverse_threshold_distance_and_ties() {
    for c in fixture()["geometry"].as_array().unwrap() {
        assert_eq!(
            table_match::geometry(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_raw_html_chunks_bold_and_logic_quirks() {
    for c in fixture()["assembled"].as_array().unwrap() {
        assert_eq!(
            table_match::assemble(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_ocr_format_filter_match_and_model_structures() {
    let f = fixture();
    for c in f["runs"].as_array().unwrap() {
        assert_eq!(
            table_match::run(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
    for c in f["real"].as_array().unwrap() {
        assert_eq!(table_match::run(&json!({"structure":c["structure"],"ocr_result":c["ocr_result"],"width":c["width"],"height":c["height"]})).unwrap(),c["expected"]);
    }
}
#[test]
fn matcher_resources_and_invalid_columns_are_explicit() {
    let f = fixture();
    let req = &f["runs"][0]["request"];
    let mut bad = req.clone();
    bad["ocr_result"][2] = json!([]);
    let before = bad.clone();
    assert!(table_match::run(&bad).is_err());
    assert_eq!(bad, before);
    bad = req.clone();
    bad["structure"]["cell_bboxes"] = json!([]);
    assert!(table_match::run(&bad).is_err());
    bad = req.clone();
    bad["ocr_result"] = json!([[], [], []]);
    assert!(table_match::run(&bad).is_err());
    let req =
        json!({"tokens":["<tr>","<td"," colspan=\"1001\"",">","</td>"],"texts":[],"matched":{}});
    assert!(table_match::assemble(&req).is_err());
    let req = json!({"tokens":["<td></td>","<td></td>","<td></td>"],"texts":["x".repeat(1_500_000)],"matched":{"0":[0],"1":[0],"2":[0]}});
    assert!(table_match::assemble(&req).is_err());
}
