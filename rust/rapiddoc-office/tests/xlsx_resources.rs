use rapiddoc_office::xlsx;
use serde_json::{json, Value};
use std::io::Read;
#[path = "support/image_parity.rs"]
mod image_parity;
use image_parity::normalize;
#[test]
fn original_xlsx_full_resources_and_order() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/xlsx_resources.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let fixture: Value = serde_json::from_slice(&raw).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let bytes: std::sync::Arc<[u8]> = c["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
            .into();
        for q in c["queries"].as_array().unwrap() {
            let actual = xlsx::Workbook::open(bytes.clone()).and_then(|w| {
                w.into_pages(
                    q["include_hidden"].as_bool().unwrap(),
                    q["gap"].as_u64().map(|v| v as u32),
                    q["singleton"].as_bool().unwrap(),
                )
            });
            if q.get("error").is_some() {
                assert!(
                    actual.is_err(),
                    "{} expected error {}",
                    c["name"],
                    q["error"]
                );
                continue;
            }
            let mut actual = json!(actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"])));
            normalize(&mut actual);
            if actual != q["pages"] {
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/xlsx-resource-diff.json");
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&json!({"name":c["name"],"query":q,"actual":actual}))
                        .unwrap(),
                )
                .unwrap();
                panic!(
                    "{} resources differ; build/xlsx-resource-diff.json",
                    c["name"]
                );
            }
        }
    }
}
