use rapiddoc_office::equation::image_mtef;
use serde_json::{json, Value};
use std::io::Read;
#[test]
fn current_python_wmf_gif_equations_cache_and_budget() {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/image_mtef.json.gz")[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    let fixture: Value = serde_json::from_slice(&bytes).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let payload: Vec<_> = case["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect();
        let mut decoder = image_mtef::Decoder::default();
        let actual = decoder.decode(&payload, case["part"].as_str(), case["mime"].as_str());
        if case.get("error").is_some() {
            assert_eq!(
                actual.unwrap_err().kind.code(),
                case["error"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        } else {
            let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
            assert_eq!(json!(actual), case["latex"], "{}", case["name"]);
            assert_eq!(
                decoder
                    .decode(&payload, case["part"].as_str(), case["mime"].as_str())
                    .unwrap(),
                actual
            );
        }
        assert_eq!(
            decoder.total_bytes as u64,
            case["total_bytes"].as_u64().unwrap(),
            "{} bytes",
            case["name"]
        );
        assert_eq!(
            decoder.warnings.len() as u64,
            case["warnings"].as_u64().unwrap(),
            "{} warnings",
            case["name"]
        );
    }
}
