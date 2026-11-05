use rapiddoc_office::{document_output as output, spreadsheet, xlsx};
use serde_json::Value;
use std::io::Read;
#[path = "support/output_parity.rs"]
mod output_parity;
fn frozen(raw: &[u8]) -> Value {
    let mut b = vec![];
    flate2::read::GzDecoder::new(raw)
        .read_to_end(&mut b)
        .unwrap();
    serde_json::from_slice(&b).unwrap()
}
#[test]
fn original_xlsx_complete_middle_assets_and_exports() {
    let reference = frozen(include_bytes!(
        "../../../tests/fixtures/office_rust_reference_068/xlsx_output.json.gz"
    ));
    let sources = [
        (
            "xlsx_cells.json.gz",
            frozen(include_bytes!(
                "../../../tests/fixtures/office_rust_reference_068/xlsx_cells.json.gz"
            )),
        ),
        (
            "xlsx_resources.json.gz",
            frozen(include_bytes!(
                "../../../tests/fixtures/office_rust_reference_068/xlsx_resources.json.gz"
            )),
        ),
        (
            "xlsx_compat.json.gz",
            frozen(include_bytes!(
                "../../../tests/fixtures/office_rust_reference_068/xlsx_compat.json.gz"
            )),
        ),
    ];
    for c in reference["cases"].as_array().unwrap() {
        let fixture = &sources
            .iter()
            .find(|(name, _)| *name == c["fixture"].as_str().unwrap())
            .unwrap()
            .1;
        let s = &fixture["cases"][c["source_index"].as_u64().unwrap() as usize];
        let bytes = s["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let parsed = xlsx::convert(bytes.into(), false, None, true)
            .and_then(|b| spreadsheet::into_body_pages(b.pages))
            .and_then(|p| output::middle(p, reference["version"].as_str().unwrap()));
        if c.get("native_recovery").is_some() {
            assert!(parsed.is_ok(), "{} native recovery", c["name"]);
            continue;
        }
        if c.get("error").is_some() {
            assert!(parsed.is_err(), "{} expected error", c["name"]);
            continue;
        }
        output_parity::verify(parsed.unwrap_or_else(|e| panic!("{}: {e}", c["name"])), c);
    }
}
