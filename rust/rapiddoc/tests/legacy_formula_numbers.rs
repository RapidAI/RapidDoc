use rapiddoc::legacy_page::{assemble_many_with_formulas, optimize_formula_numbers};
use rapiddoc_core::ErrorCode;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-formula-numbers.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
#[test]
fn complete_formula_number_fields_match_old_python() {
    let fixture = fixture();
    for c in fixture["cases"].as_array().unwrap() {
        let input = c["pages"].clone();
        assert_eq!(
            optimize_formula_numbers(&input).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
        assert_eq!(input, c["pages"]);
    }
}
#[test]
fn models_formula_numbers_and_paragraphs_match_old_stages() {
    let fixture = fixture();
    for c in fixture["chains"].as_array().unwrap() {
        assert_eq!(
            assemble_many_with_formulas(&c["requests"]).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}
#[test]
fn invalid_input_and_resource_limits_are_errors() {
    assert_eq!(
        optimize_formula_numbers(&json!({})).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        optimize_formula_numbers(&Value::Array(vec![json!({}); 1001]))
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        optimize_formula_numbers(&json!([{"meta":"x".repeat(2_000_001)}]))
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(optimize_formula_numbers(&json!([{"preproc_blocks":[{"type":"formula_number","bbox":[0,0,10,10],"lines":[{"spans":[{"type":"text","content":null}]}]}]}])).unwrap_err().code,ErrorCode::InvalidInput);
}
