use serde_json::Value;
#[test]
fn old_input_geometry_overlap_count_and_vector_diagnostic_guards() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-engine-policy-stable.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_engine_policy::inspect(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        )
    }
}
