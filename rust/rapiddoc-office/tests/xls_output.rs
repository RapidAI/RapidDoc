use rapiddoc_office::{document_output as output, spreadsheet, xls::Workbook};
use serde_json::Value;
use std::io::Read;
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
fn original_xls_complete_middle_assets_and_exports() {
    let f = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/xls_output.json.gz"
    ));
    let sources = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/xls_resources.json.gz"
    ));
    for c in f["cases"].as_array().unwrap() {
        let s = &sources["cases"][c["source_index"].as_u64().unwrap() as usize];
        let bytes = s["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let actual = Workbook::open_complete(bytes.into())
            .and_then(Workbook::into_pages)
            .and_then(spreadsheet::into_body_pages)
            .and_then(|p| output::middle(p, f["version"].as_str().unwrap()));
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected error", c["name"]);
            continue;
        }
        output_parity::verify(actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"])), c);
    }
}
