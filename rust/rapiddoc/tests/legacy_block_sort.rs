use rapiddoc::legacy_block_sort::sort;
use rapiddoc_core::ErrorCode;
use serde_json::{json, Value};

#[test]
fn complete_model_order_pipeline_matches_frozen_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-block-sort.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(fixture["cases"].as_array().unwrap().len() > 1000);
    for c in fixture["cases"].as_array().unwrap() {
        let before = c["request"].clone();
        assert_eq!(sort(&before).unwrap(), c["expected"], "{}", c["name"]);
        assert_eq!(before, c["request"]);
    }
}

#[test]
fn old_sort_and_paragraph_chain_matches_including_cpu_ocr() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-sort-paragraphs.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 308);
    for c in fixture["cases"].as_array().unwrap() {
        let r = &c["request"];
        let blocks = sort(r).unwrap();
        let pages = json!([{"page_idx":0,"page_size":[r["page_width"],r["page_height"]],"preproc_blocks":blocks}]);
        assert_eq!(
            rapiddoc::legacy_paragraphs::split(&pages).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}

#[test]
fn missing_model_order_and_invalid_or_excessive_inputs_fail_explicitly() {
    let b = json!({"type":"text","bbox":[0,0,100,20],"lines":[],"original_order":0});
    let req = json!({"blocks":[b],"page_width":600,"page_height":800,"footnote_blocks":[]});
    let mut no_order = req.clone();
    no_order["blocks"][0]["original_order"] = Value::Null;
    assert_eq!(sort(&no_order).unwrap_err().code, ErrorCode::Unsupported);
    let mut bad = req.clone();
    bad["blocks"][0]["bbox"] = json!([0, 0, 0, 20]);
    assert_eq!(sort(&bad).unwrap_err().code, ErrorCode::InvalidInput);
    let mut too_many = req.clone();
    too_many["blocks"] = Value::Array(vec![req["blocks"][0].clone(); 1001]);
    assert_eq!(sort(&too_many).unwrap_err().code, ErrorCode::ResourceLimit);
    let mut huge = req.clone();
    huge["metadata"] = json!("x".repeat(2_000_001));
    assert_eq!(sort(&huge).unwrap_err().code, ErrorCode::ResourceLimit);
}
