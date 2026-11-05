use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-text.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_character_selection_normalization_and_local_geometry() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_text::pending(&c["request"]).unwrap(),
            c["pending"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_complete_native_glyph_rows_tokens_and_advance_geometry() {
    for (i, c) in fixture()["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_text::build(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
    let extra: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-text-reversed.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in extra["cases"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_text::build(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_cell_visual_refinement_text_and_source_parts() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-cell-text.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_text::cell_parts(&c["request"]).unwrap(),
            c["expected"],
            "case {i}: {}",
            c["request"]
        );
    }
}
#[test]
fn old_cell_latin_url_unit_and_hyphen_row_separators() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-cell-text.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in f["separators"].as_array().unwrap() {
        let p = c["previous"].as_str().unwrap();
        assert_eq!(
            rapiddoc::legacy_model::table_native_text::row_separator(
                p,
                p,
                c["next"].as_str().unwrap()
            ),
            c["expected"].as_str().unwrap(),
            "{c}"
        );
    }
}
