use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-aligned-numeric.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_regroup_numeric_groups_and_complete_aligned_candidates() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_aligned_numeric::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_unicode_decimal_numeric_token_regex() {
    for c in fixture()["regex"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_aligned_numeric::numeric_token(
                c["text"].as_str().unwrap()
            ),
            c["expected"].as_bool().unwrap(),
            "{}",
            c["text"]
        );
    }
}
