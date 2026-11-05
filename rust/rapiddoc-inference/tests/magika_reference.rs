//! Run explicitly against frozen original Python features/labels and local ORT.
#[test]
#[ignore = "requires MAGIKA_REFERENCE and RAPIDDOC_ONNXRUNTIME_LIBRARY"]
fn exact_original_features_and_language_labels() {
    let source = std::env::var("MAGIKA_REFERENCE").expect("frozen Python reference");
    let runtime = std::env::var("RAPIDDOC_ONNXRUNTIME_LIBRARY").expect("ORT library");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
    let cases = reference["languages"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            serde_json::json!(rapiddoc_inference::magika::features(text.as_bytes())),
            case["features"]
        );
        let label =
            rapiddoc_inference::magika::language(text, Some(std::path::Path::new(&runtime)))
                .unwrap();
        assert_eq!(label, case["label"].as_str().unwrap(), "{text:?}");
    }
}
