use serde_json::Value;
fn fixture() -> Value {
    let mut f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-rule-band.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let extra: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-rule-band-extra.json"),
        )
        .unwrap(),
    )
    .unwrap();
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    f
}
#[test]
fn old_physical_header_band_candidates_and_descriptor_rowspans() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_rule_band::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_shared_sparse_column_hypotheses_and_gutter_diagnostics() {
    for (i, c) in fixture()["columns"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_rule_band::inspect_columns(&c["request"]).unwrap(),
            c["expected"],
            "columns {i}: {}",
            c["request"]
        );
    }
}
