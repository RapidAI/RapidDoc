use rapiddoc_inference::formula::{repair, Decoder};
use serde_json::Value;
fn fixture() -> (Value, Decoder) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/formula-decode.json")).unwrap(),
    )
    .unwrap();
    let d = Decoder::new(
        &std::fs::read_to_string(root.join("build/migration/formula-character.json")).unwrap(),
    )
    .unwrap();
    (f, d)
}
#[test]
fn actual_old_formula_repairs_and_tokens() {
    let (f, d) = fixture();
    for (i, c) in f["repairs"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            repair(c["text"].as_str().unwrap()).unwrap(),
            c["expected"],
            "repair {i}"
        );
    }
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        let row = serde_json::from_value(c["tokens"].clone()).unwrap();
        let out = d.decode(&[row]);
        assert_eq!(out.unwrap()[0], c["expected"], "decode {i}");
    }
}
#[test]
fn formula_decode_limits() {
    let (_, d) = fixture();
    assert!(repair(&"a".repeat(65_537)).is_err());
    assert!(d.decode(&[vec![0; 8193]]).is_err());
    assert!(Decoder::new("{}").is_err());
    assert!(Decoder::new(&" ".repeat(8_000_001)).is_err());
}

#[test]
fn actual_old_extra_token_configurations() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let original: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/formula-character.json")).unwrap(),
    )
    .unwrap();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/formula-token-configs-v2.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let mut data = original.clone();
        data["tokenizer_config_file"] = c["config"].clone();
        let decoder = Decoder::new(&data.to_string()).unwrap();
        let rows: Vec<Vec<u32>> = serde_json::from_value(c["rows"].clone()).unwrap();
        assert_eq!(
            serde_json::to_value(decoder.decode(&rows).unwrap()).unwrap(),
            c["expected"]
        );
    }
}
