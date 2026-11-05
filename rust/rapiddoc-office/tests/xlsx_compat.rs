use rapiddoc_office::{package::Package, xlsx};
use serde_json::{json, Value};
use std::io::Read;
#[path = "support/image_parity.rs"]
mod image_parity;
fn tree(n: roxmltree::Node<'_, '_>) -> Value {
    fn name(ns: Option<&str>, local: &str) -> String {
        if let Some(ns) = ns {
            format!("{{{ns}}}{local}")
        } else {
            local.into()
        }
    }
    let attrs = n
        .attributes()
        .map(|a| (name(a.namespace(), a.name()), json!(a.value())))
        .collect::<serde_json::Map<_, _>>();
    json!([
        name(n.tag_name().namespace(), n.tag_name().name()),
        attrs,
        n.children()
            .next()
            .filter(|n| n.is_text())
            .and_then(|n| n.text()),
        n.children()
            .filter(|n| n.is_element())
            .map(tree)
            .collect::<Vec<_>>()
    ])
}
#[test]
fn original_xlsx_package_normalization_and_retry() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/xlsx_compat.json.gz")[..]).read_to_end(&mut raw).unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes: std::sync::Arc<[u8]> = c["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
            .into();
        if let Some(expected) = c["normalized"].as_object() {
            let mut p = Package::open(bytes.clone()).unwrap();
            let names = xlsx::normalize::apply(&mut p).unwrap();
            assert_eq!(
                names.len(),
                expected.len(),
                "{} normalized count",
                c["name"]
            );
            for n in names {
                let s = p.read_xml(&n).unwrap().unwrap();
                let t = rapiddoc_office::xml::parse(&s).unwrap();
                assert_eq!(
                    tree(t.root_element()),
                    expected[&n],
                    "{} normalized {n}",
                    c["name"]
                );
            }
        }
        let actual = xlsx::convert(bytes, false, None, true);
        if let Some(expected) = c.get("native_recovery_pages") {
            let mut pages = json!(
                actual
                    .unwrap_or_else(|e| panic!("{} recovery: {e}", c["name"]))
                    .pages
            );
            image_parity::normalize(&mut pages);
            assert_eq!(&pages, expected, "{} native recovery", c["name"]);
            continue;
        }
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected {}", c["name"], c["error"]);
            continue;
        }
        let mut pages = json!(
            actual
                .unwrap_or_else(|e| panic!("{}: {e}", c["name"]))
                .pages
        );
        image_parity::normalize(&mut pages);
        assert_eq!(pages, c["pages"], "{} retry output", c["name"]);
    }
}
