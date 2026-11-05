use rapiddoc_office::{equation::omml, xml};
use serde_json::Value;
use std::io::Read;
fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/omml.json.gz")[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
#[test]
fn original_python_omml_structures_and_errors() {
    for case in fixture()["cases"].as_array().unwrap() {
        let document = xml::parse(case["xml"].as_str().unwrap()).unwrap();
        let actual = omml::raw(document.root_element());
        if case.get("error").is_some() {
            assert!(
                actual.is_err(),
                "{} expected {} got {actual:?}",
                case["name"],
                case["error"]
            );
        } else {
            assert_eq!(
                actual.unwrap_or_else(|e| panic!("{}: {e}", case["name"])),
                case["raw"].as_str().unwrap(),
                "{}",
                case["name"]
            );
            assert_eq!(
                omml::convert(document.root_element()).unwrap(),
                case["latex"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}
#[test]
fn full_non_identity_character_mapping_matches_pylatexenc() {
    for case in fixture()["unicode"].as_array().unwrap() {
        let char = case["char"].as_str().unwrap().chars().next().unwrap();
        assert_eq!(
            omml::unicode(char),
            case["latex"].as_str().unwrap(),
            "U+{:04X}",
            char as u32
        );
    }
    // Unlisted scalars are kept, including CJK, newly assigned characters and
    // the final Unicode scalar, rather than forced into guessed math commands.
    for char in ['中', '🌟', '\u{10ffff}'] {
        assert_eq!(omml::unicode(char), char.to_string());
    }
}
