use rapiddoc_office::chart;
use serde_json::{json, Value};
use std::{io::Read, sync::Arc};
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_chart_specs_workbooks_cache_and_fallback() {
    let mut data = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/office_charts.json.gz")
            [..],
    )
    .read_to_end(&mut data)
    .unwrap();
    let f: Value = serde_json::from_slice(&data).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let chart = bytes(c["chart_hex"].as_str().unwrap());
        let wb = c["workbook_hex"]
            .as_str()
            .map(|s| Arc::<[u8]>::from(bytes(s)));
        let spec = chart::parse(&chart);
        assert_eq!(
            serde_json::to_value(&spec).unwrap(),
            c["spec"],
            "{} spec",
            c["name"]
        );
        check(
            spec.as_ref()
                .map(|s| chart::render(s, None))
                .unwrap_or_else(|| Ok(String::new())),
            c,
            "cache",
        );
        if let Some(ref bytes) = wb {
            let workbook = chart::workbook::Workbook::open(Arc::clone(bytes));
            check(
                workbook
                    .as_ref()
                    .ok()
                    .map(|w| w.html())
                    .unwrap_or_else(|| Ok(String::new())),
                c,
                "fallback",
            );
            check(
                spec.as_ref()
                    .zip(workbook.as_ref().ok())
                    .map(|(s, w)| chart::render(s, Some(w)))
                    .unwrap_or_else(|| Ok(String::new())),
                c,
                "workbook",
            );
        }
        check(chart::extract(&chart, wb), c, "html");
    }
    for c in f["serials"].as_array().unwrap() {
        let v = match c["value"].as_str().unwrap() {
            "inf" => f64::INFINITY,
            "nan" => f64::NAN,
            s => s.parse().unwrap(),
        };
        assert_eq!(
            json!(chart::value::serial(v, c["epoch"].as_bool().unwrap())),
            c["result"],
            "{c}"
        );
    }
}

fn check(actual: rapiddoc_office::Result<String>, case: &Value, key: &str) {
    if case.get(format!("{key}_error")).is_some() {
        assert!(actual.is_err(), "{} {key} expected error", case["name"]);
    } else {
        assert_eq!(
            actual.unwrap_or_else(|e| panic!("{} {key}: {e}", case["name"])),
            case[key],
            "{} {key}",
            case["name"]
        );
    }
}
