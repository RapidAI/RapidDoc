#[path = "support/output_parity.rs"]
mod output_parity;
use rapiddoc_office::{doc::body, ole::OleReader, document_output as output};
use std::io::Read;
#[test]
fn pinned_poi_real_doc_middle_pixels_and_four_exports() {
    let input =
        include_bytes!("../../../tests/fixtures/office_rust_reference_068/doc_external.json.gz");
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(&input[..])
        .read_to_end(&mut bytes)
        .unwrap();
    let fixture: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let data: Vec<u8> = c["source_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let mut ole = OleReader::open(data.into()).unwrap();
        let body = body::convert(&mut ole).unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let parsed = output::middle(body.pages, fixture["version"].as_str().unwrap()).unwrap();
        output_parity::verify(parsed, c);
    }
}
