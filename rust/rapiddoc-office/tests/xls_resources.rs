use rapiddoc_office::{officeart, xls::records::Budget, xls::Workbook};
use serde_json::{json, Value};
use std::io::Read;
#[path = "support/image_parity.rs"]
mod image_parity;
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_xls_resource_binding_and_complete_page_order() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/xls_resources.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let actual = Workbook::open_complete(bytes(c["package_hex"].as_str().unwrap()).into());
        if c.get("error").is_some() && c.get("sheets").is_none() {
            assert!(actual.is_err(), "{} expected error", c["name"]);
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let mut sheets=json!(actual.sheets.iter().map(|s|json!({"name":s.name,"cells":s.cells.values().collect::<Vec<_>>(),"images":s.images,"equations":s.equations,"charts":s.charts})).collect::<Vec<_>>());
        image_parity::normalize(&mut sheets);
        let result = actual.into_pages();
        let result = if c.get("error").is_some() {
            assert!(result.is_err(), "{} expected page error", c["name"]);
            continue;
        } else {
            result.unwrap_or_else(|e| panic!("{} pages: {e}", c["name"]))
        };
        let mut pages = json!(result);
        image_parity::normalize(&mut pages);
        if sheets != c["sheets"] || pages != c["pages"] {
            std::fs::write(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/xls-resource-diff.json"),serde_json::to_vec_pretty(&json!({"name":c["name"],"actual_sheets":sheets,"expected_sheets":c["sheets"],"actual_pages":pages,"expected_pages":c["pages"]})).unwrap()).unwrap();
            panic!(
                "{} resources differ; build/xls-resource-diff.json",
                c["name"]
            );
        }
    }
    for c in f["officeart"].as_array().unwrap() {
        let data = bytes(c["hex"].as_str().unwrap());
        let mut budget = Budget::default();
        if c.get("shapes").is_some() || c.get("error").is_some() {
            let actual = officeart::excel_shapes(&data, &mut budget);
            if c.get("error").is_some() {
                assert!(actual.is_err(), "{}", c["name"])
            } else {
                assert_eq!(json!(actual.unwrap()), c["shapes"], "{}", c["name"]);
            }
        } else {
            let delay = c["delay_hex"].as_str().map(bytes);
            let actual =
                officeart::bstore(&data, delay.as_deref(), &mut budget, &mut vec![]).unwrap();
            let actual=actual.iter().map(|(i,p)|(i.to_string(),json!({"data_hex":p.data.iter().map(|b|format!("{b:02x}")).collect::<String>(),"extension":p.extension,"content_type":p.content_type,"render_size_emu":p.render_size_emu}))).collect::<serde_json::Map<_,_>>();
            assert_eq!(json!(actual), c["images"], "{}", c["name"]);
        }
        assert_eq!(
            json!(budget.count),
            c["visits"],
            "{} record charges",
            c["name"]
        );
    }
}
