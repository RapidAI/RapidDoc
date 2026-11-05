use serde_json::Value;
fn fixture() -> Value {
    let mut f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-candidate.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let extra: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-candidate-extra.json"),
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
fn old_complete_grid_validation_index_and_glyph_ownership() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        let result =
            rapiddoc::legacy_model::table_native_candidate::inspect(&c["request"]).unwrap();
        for key in ["valid", "index", "assignments"] {
            assert_eq!(result[key], c[key], "case {i}/{key}: {}", c["request"]);
        }
    }
}
#[test]
fn old_candidate_rejection_content_provenance_and_quality() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_candidate::build(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_plain_candidate_html_escaping_spans_and_empty_rows() {
    for c in fixture()["html"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_candidate::render(c).unwrap(),
            c["expected"].as_str().unwrap()
        );
    }
}
