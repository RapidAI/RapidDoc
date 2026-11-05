use serde_json::Value;
#[test]
fn old_complete_multiline_candidates_diagnostics_and_html() {
    let mut f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-multiline.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let extra: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-multiline-extra.json"),
        )
        .unwrap(),
    )
    .unwrap();
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_multiline::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["label"]
        );
    }
}
