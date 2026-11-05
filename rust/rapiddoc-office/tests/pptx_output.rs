use rapiddoc_office::{document_output as output, package::Package};
use serde_json::Value;
use std::io::Read;
#[path = "support/image_parity.rs"]
mod image_parity;
#[path = "support/output_parity.rs"]
mod output_parity;
fn frozen(data: &[u8]) -> Value {
    let mut b = vec![];
    flate2::read::GzDecoder::new(data)
        .read_to_end(&mut b)
        .unwrap();
    serde_json::from_slice(&b).unwrap()
}
#[test]
fn original_pptx_complete_pages_middle_assets_and_exports() {
    let f = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/pptx_output.json.gz"
    ));
    let sources = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/pptx_context.json.gz"
    ));
    for c in f["cases"].as_array().unwrap() {
        let s = &sources["cases"][c["source_index"].as_u64().unwrap() as usize];
        let bytes = base64::decode(s["package_b64"].as_str().unwrap()).unwrap();
        let actual = Package::open(bytes.into()).and_then(rapiddoc_office::pptx::convert_package);
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected {}", c["name"], c["error"]);
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let mut pages = serde_json::json!(actual.pages);
        let mut expected = c["pages"].clone();
        image_parity::normalize(&mut pages);
        image_parity::normalize(&mut expected);
        assert_eq!(pages, expected, "{} pages", c["name"]);
        let result = output::middle(actual.pages, f["version"].as_str().unwrap()).unwrap();
        output_parity::verify(result, c);
        if s["reference_normalized_input"] == true {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(s["name"].as_str().unwrap());
            let raw = std::fs::read(path).unwrap();
            let body =
                rapiddoc_office::pptx::convert_package(Package::open(raw.into()).unwrap()).unwrap();
            output_parity::verify(
                output::middle(body.pages, f["version"].as_str().unwrap()).unwrap(),
                c,
            );
        }
    }
}
