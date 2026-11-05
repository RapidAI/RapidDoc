use serde_json::Value;
#[test]
#[ignore = "requires captured local real-table fixtures in build/model-switch-real-raster.json"]
fn real_raster_tables_match_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/model-switch-real-raster.json");
    let cases: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let png = std::fs::read(c["image"].as_str().unwrap()).unwrap();
        let result = rapiddoc::legacy_model::table_ocr_rule::recover(&png, &c["ocr"]).unwrap();
        assert_eq!(
            !result.is_null(),
            c["accepted"].as_bool().unwrap(),
            "{}",
            c["image"]
        );
        if !result.is_null() {
            assert_eq!(result["html"], c["html"], "{}", c["image"]);
        }
    }
}
#[test]
fn raster_rule_recovery_matches_python_and_falls_back_on_uncertain_words() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/model_switching");
    let cases: Value =
        serde_json::from_slice(&std::fs::read(root.join("raster.json")).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let png = std::fs::read(root.join(c["image"].as_str().unwrap())).unwrap();
        let actual = rapiddoc::legacy_model::table_ocr_rule::recover(&png, &c["ocr"]).unwrap();
        assert_eq!(
            !actual.is_null(),
            c["accepted"].as_bool().unwrap(),
            "{}",
            c["image"]
        );
        if !actual.is_null() {
            assert_eq!(actual["html"], c["html"]);
        }
        let mut low = c["ocr"].clone();
        low[2][0] = serde_json::json!(0.3);
        assert!(rapiddoc::legacy_model::table_ocr_rule::recover(&png, &low)
            .unwrap()
            .is_null());
    }
}
