use serde_json::Value;
#[test]
fn old_raw_html_selection_and_metrics() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/legacy-table-routing.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in fixture["selection"].as_array().unwrap() {
        let result = rapiddoc::legacy_model::table_selection::select(c).unwrap();
        for k in ["html", "wired", "wireless"] {
            assert_eq!(result[k], c[k], "{c}");
        }
    }
}
