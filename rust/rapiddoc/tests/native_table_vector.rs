use serde_json::Value;
#[test]
fn old_canonical_tracks_alias_conflicts_and_physical_double_boundaries() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-tracks.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::canonicalize(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_equal_distance_physical_rules_use_python_set_iteration() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-track-ties.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::canonicalize(&c["request"]).unwrap(),
            c["expected"],
            "tie case {i}"
        );
    }
}
#[test]
fn old_physical_fragments_merge_and_raw_supported_tracks() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-vector.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::prepare(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
