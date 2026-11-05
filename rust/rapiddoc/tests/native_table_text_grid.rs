use serde_json::Value;
#[test]
fn old_sparse_wireless_key_value_candidates_and_diagnostics() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-text-grid.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_text_grid::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        )
    }
}
