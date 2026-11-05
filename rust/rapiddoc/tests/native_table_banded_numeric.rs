use serde_json::Value;
#[test]
fn old_banded_numeric_candidates_and_reprojected_cell_text() {
    let mut f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-banded-numeric.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let extra: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-banded-numeric-extra.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in extra["errors"].as_array().unwrap() {
        assert!(
            rapiddoc::legacy_model::table_native_banded_numeric::generate(&c["request"]).is_err()
        );
    }
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    let ties: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-banded-numeric-ties.json"),
        )
        .unwrap(),
    )
    .unwrap();
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(ties["cases"].as_array().unwrap().iter().cloned());
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_banded_numeric::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["label"]
        );
    }
}
