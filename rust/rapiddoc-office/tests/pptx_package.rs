use rapiddoc_office::{package::Package, pptx, xml};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
fn canonical(n: roxmltree::Node<'_, '_>) -> Value {
    let name = |ns: Option<&str>, local: &str| json!([ns.unwrap_or(""), local]);
    let mut attributes = n
        .attributes()
        .map(|a| json!([name(a.namespace(), a.name()), a.value()]))
        .collect::<Vec<_>>();
    attributes.sort_by_key(|v| v.to_string());
    let text = n
        .children()
        .take_while(|c| !c.is_element())
        .filter_map(|c| c.text())
        .collect::<String>();
    let text = if !text.trim().is_empty() || n.tag_name().name() == "t" {
        text
    } else {
        String::new()
    };
    let children = n
        .children()
        .filter(|c| c.is_element())
        .map(|c| {
            let tail = c
                .next_siblings()
                .skip(1)
                .take_while(|n| !n.is_element())
                .filter_map(|n| n.text())
                .collect::<String>();
            json!([
                canonical(c),
                if tail.trim().is_empty() { "" } else { &tail }
            ])
        })
        .collect::<Vec<_>>();
    json!([
        name(n.tag_name().namespace(), n.tag_name().name()),
        attributes,
        text,
        children
    ])
}
#[test]
fn original_pptx_normalization_members_and_errors() {
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/pptx_package.json.gz")[..]).read_to_end(&mut bytes).unwrap();
    let f: Value = serde_json::from_slice(&bytes).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let mut package = Package::open(
            base64::decode(c["package_b64"].as_str().unwrap())
                .unwrap()
                .into(),
        )
        .unwrap();
        let result = pptx::normalize::apply(&mut package);
        if c.get("error").is_some() {
            assert!(result.is_err(), "{} expected {}", c["name"], c["error"]);
            continue;
        }
        let changes = result.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        assert_eq!(
            !changes.is_empty(),
            c["changed"].as_bool().unwrap(),
            "{} changed",
            c["name"]
        );
        assert_eq!(
            package.names().len(),
            c["members"].as_object().unwrap().len(),
            "{} members",
            c["name"]
        );
        for (name, expected) in c["members"].as_object().unwrap() {
            let bytes = package.required(name).unwrap();
            let actual = if expected.get("xml").is_some() {
                let text = xml::decode(&bytes).unwrap();
                let doc = xml::parse(&text).unwrap();
                json!({"xml":canonical(doc.root_element())})
            } else {
                json!({"sha256":Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>()})
            };
            assert_eq!(actual, *expected, "{} {name}", c["name"]);
        }
    }
}

#[path = "support/image_parity.rs"]
mod image_parity;
#[test]
fn original_pptx_failed_package_retry_output() {
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/pptx_package.json.gz")[..]).read_to_end(&mut bytes).unwrap();
    let f: Value = serde_json::from_slice(&bytes).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let package = Package::open(
            base64::decode(c["package_b64"].as_str().unwrap())
                .unwrap()
                .into(),
        )
        .unwrap();
        let actual = pptx::convert_package(package);
        if c.get("conversion_error").is_some() {
            assert!(
                actual.is_err(),
                "{} expected {}",
                c["name"],
                c["conversion_error"]
            );
            continue;
        }
        let mut actual = json!(
            actual
                .unwrap_or_else(|e| panic!("{}: {e}", c["name"]))
                .pages
        );
        let mut expected = c["pages"].clone();
        image_parity::normalize(&mut actual);
        image_parity::normalize(&mut expected);
        assert_eq!(actual, expected, "{} retry pages", c["name"]);
    }
}
