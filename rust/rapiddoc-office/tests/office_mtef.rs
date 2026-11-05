use rapiddoc_office::equation::mtef;
use serde_json::{json, Value};
use std::io::Read;
#[test]
fn current_python_v3_v5_native_and_ole_equations() {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/mtef.json.gz")[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    let fixture: Value = serde_json::from_slice(&bytes).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let hex = case["hex"].as_str().unwrap();
        let payload: Vec<_> = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect();
        if case["kind"] == "pool" {
            let mut ole = rapiddoc_office::ole::OleReader::open(payload.into()).unwrap();
            assert_eq!(
                json!(mtef::object_pool(&mut ole).unwrap()),
                case["latex"],
                "{}",
                case["name"]
            );
            continue;
        }
        let actual = match case["kind"].as_str().unwrap() {
            "mtef" => mtef::decode(&payload),
            "native" => mtef::equation_native(&payload),
            "object" => mtef::equation_object(payload.into()),
            _ => unreachable!(),
        };
        if case.get("error").is_some() {
            assert_eq!(
                actual.unwrap_err().kind.code(),
                case["error"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        } else {
            assert_eq!(
                json!(actual.unwrap_or_else(|e| panic!("{}: {e}", case["name"]))),
                case["latex"],
                "{}",
                case["name"]
            );
        }
    }
}
