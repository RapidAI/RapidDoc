use rapiddoc::legacy_model::table_normalize;
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/table-normalize.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_cjk_text_boundaries_and_unicode_whitespace() {
    for c in fixture()["texts"].as_array().unwrap() {
        assert_eq!(
            table_normalize::text(c["input"].as_str().unwrap()).unwrap(),
            c["expected"].as_str().unwrap(),
            "{}",
            c["input"]
        );
    }
}
#[test]
fn old_html_entities_children_identity_and_serialization() {
    let f = fixture();
    let mut bad = Vec::new();
    for c in f["html"].as_array().unwrap() {
        let result = table_normalize::html(c["input"].as_str().unwrap());
        if c["raises"] == true {
            if result.is_ok() {
                bad.push(format!("expected error: {}", c["input"]));
            }
            continue;
        }
        if result.as_ref().map(String::as_str).ok() != c["expected"].as_str() {
            bad.push(format!(
                "{} => {:?}; expected {}",
                c["input"], result, c["expected"]
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} mismatches: {:?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
}
#[test]
fn actual_slanet_outputs_final_html_match_old() {
    for c in fixture()["real"].as_array().unwrap() {
        assert_eq!(
            table_normalize::html(c["raw_html"].as_str().unwrap()).unwrap(),
            c["expected"].as_str().unwrap(),
            "{}",
            c["variant"]
        );
    }
}
