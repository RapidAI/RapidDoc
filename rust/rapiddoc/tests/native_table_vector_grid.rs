use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-vector-grid.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_topology_physical_evidence_and_materialized_candidates() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::build(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_statistics_mean_for_separator_decisions() {
    for c in fixture()["means"].as_array().unwrap() {
        let values: Vec<f64> = serde_json::from_value(c["values"].clone()).unwrap();
        assert_eq!(
            rapiddoc_native::table_decisiveness_mean(&values).unwrap(),
            c["expected"].as_f64().unwrap()
        );
    }
}
#[test]
fn old_five_percent_gate_l_shape_and_physical_blank_rows() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-vector-grid-edges.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::build(&c["request"]).unwrap(),
            c["expected"],
            "edge case {i}"
        );
    }
}
