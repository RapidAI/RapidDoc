#[path = "support/output_parity.rs"]
mod output_parity;
use rapiddoc_office::{docx, document_output as output};
use serde_json::Value;
use std::io::Read;
fn frozen(data: &[u8]) -> Value {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(data)
        .read_to_end(&mut raw)
        .unwrap();
    serde_json::from_slice(&raw).unwrap()
}
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_complete_office_middle_assets_and_exports() {
    let references = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/office_output.json.gz"
    ));
    let bodies = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/docx_body.json.gz"
    ));
    let resources = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/docx_resources.json.gz"
    ));
    for c in references["cases"].as_array().unwrap() {
        let fixture = if c["fixture"] == "docx_body.json.gz" {
            &bodies
        } else {
            &resources
        };
        let source = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["name"] == c["name"])
            .unwrap();
        let parsed = docx::Source::open(bytes(source["package_hex"].as_str().unwrap()).into())
            .and_then(docx::body::convert)
            .and_then(|b| output::middle(b.pages, references["version"].as_str().unwrap()));
        if c.get("error").is_some() {
            assert!(parsed.is_err(), "{} expected error", c["name"]);
            continue;
        }
        let parsed = parsed.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        output_parity::verify(parsed, c);
    }
}
