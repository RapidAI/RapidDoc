use serde_json::Value;
#[path = "../src/processing/table/table_native_vector/pipeline/digit.rs"]
mod digit;
#[test]
fn complete_unicode16_isdigit_matches_the_old_runtime() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-digit.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let expected: Vec<u32> = serde_json::from_value(f["isdigit"].clone()).unwrap();
    let actual: Vec<_> = (0..0x110000)
        .filter(|n| {
            char::from_u32(*n).is_some_and(|c| {
                rapiddoc_native::python_decimal(c).is_some()
                    || digit::EXTRA.binary_search(n).is_ok()
            })
        })
        .collect();
    assert_eq!(actual, expected);
}
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-vector-pipeline.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_complete_vector_candidates_and_hypothesis_diagnostics() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::generate(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_complete_track_hard_gates_and_dense_baseline_policy() {
    for (i, c) in fixture()["track_cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::inspect_tracks(&c["request"]).unwrap(),
            c["expected"],
            "tracks {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_pdf_primitives_and_large_track_grid_limits() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-vector-pdf.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_vector::generate(&c["request"]).unwrap(),
            c["expected"],
            "pdf/limit case {i}"
        );
    }
}
