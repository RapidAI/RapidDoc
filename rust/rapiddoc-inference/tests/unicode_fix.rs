use serde_json::Value;
#[test]
fn actual_ftfy_unicode_and_mojibake() {
    assert_eq!(unicode_normalization::UNICODE_VERSION, (16, 0, 0));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/unicode-fix.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(f["cases"].as_array().unwrap().len(), 29517);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc_inference::unicode_fix::fix_text(c["text"].as_str().unwrap()).unwrap(),
            c["expected"],
            "ftfy {i}: {:?}",
            c["text"]
        );
    }
}
#[test]
fn ftfy_budgets() {
    assert!(rapiddoc_inference::unicode_fix::fix_text(&"x".repeat(65537)).is_err());
    assert!(
        rapiddoc_inference::unicode_fix::fix_text(&("&amp;".to_owned() + &"amp;".repeat(200)))
            .is_err()
    );
}

#[test]
fn actual_formula_unicode_postprocessing() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/unicode-fix.json")).unwrap(),
    )
    .unwrap();
    let expected: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/formula-unicode.json")).unwrap(),
    )
    .unwrap();
    for (i, (c, e)) in source["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["expected"].as_array().unwrap())
        .enumerate()
    {
        assert_eq!(
            rapiddoc_inference::formula::post_process(c["text"].as_str().unwrap()).unwrap(),
            *e,
            "post_process {i}"
        );
    }
}
