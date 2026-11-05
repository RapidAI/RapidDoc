use rapiddoc_office::{pptx::math_fallback, xml};
use serde_json::Value;
use std::io::Read;
#[test]
fn original_fork_secondary_math_engine() {
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/pptx_math.json.gz")[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    let f: Value = serde_json::from_slice(&bytes).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let doc = xml::parse(c["xml"].as_str().unwrap()).unwrap();
        let actual = math_fallback::convert(doc.root_element());
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected {}", c["name"], c["error"]);
        } else {
            assert_eq!(
                actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"])),
                c["latex"].as_str().unwrap(),
                "{}",
                c["name"]
            );
        }
    }
}
