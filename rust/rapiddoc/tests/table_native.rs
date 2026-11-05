use rapiddoc::legacy_model::table_native;
use serde_json::{json, Value};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(root().join("build/migration/table-native.json")).unwrap(),
    )
    .unwrap()
}
#[test]
fn old_table_native_geometry() {
    for c in fixture()["geometry"].as_array().unwrap() {
        assert_eq!(
            table_native::spans(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_native_table_line_and_character_results() {
    let f = fixture();
    for c in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(f["real"].as_array().unwrap())
    {
        let png = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        assert_eq!(
            table_native::extract(&png, &c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn native_table_rejects_invalid_inputs_without_mutation() {
    let f = fixture();
    let c = &f["cases"][0];
    let png = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
    for (key, bad) in [
        ("scale", json!(0)),
        ("word_boxes", json!("yes")),
        ("boxes", json!([[[0, 0]]])),
        ("native_page", json!({})),
    ] {
        let mut req = c["request"].clone();
        req[key] = bad;
        let before = req.clone();
        assert!(table_native::extract(&png, &req).is_err());
        assert_eq!(req, before);
    }
    assert!(table_native::extract(b"bad", &c["request"]).is_err());
}
