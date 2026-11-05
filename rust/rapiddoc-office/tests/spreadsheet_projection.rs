use rapiddoc_office::spreadsheet::{self, Sheet};
use serde_json::{json, Value};
use std::io::Read;
#[test]
fn original_spreadsheet_regions_projection_merges_and_html() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/spreadsheet_projection.json.gz"
        )[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let mut sheet = Sheet::new(
            serde_json::from_value(c["cells"].clone()).unwrap(),
            serde_json::from_value(c["merges"].clone()).unwrap(),
        )
        .unwrap();
        sheet.default_cell = serde_json::from_value(c["default_cell"].clone()).unwrap();
        assert_eq!(json!(sheet.bounds()), c["bounds"], "{} bounds", c["name"]);
        for q in c["queries"].as_array().unwrap() {
            let gap = q["gap"].as_u64().map(|v| v as u32);
            let (selected, penalty, tables) = sheet.project(gap);
            assert_eq!(
                selected,
                q["selected"].as_u64().unwrap() as u32,
                "{} gap",
                c["name"]
            );
            assert!(
                (penalty - q["penalty"].as_f64().unwrap()).abs() < 1e-12,
                "{} penalty",
                c["name"]
            );
            assert_eq!(
                serde_json::to_value(&tables).unwrap(),
                q["tables"],
                "{} {:?} table IR",
                c["name"],
                gap
            );
            assert_eq!(
                json!(tables
                    .iter()
                    .map(spreadsheet::table::html)
                    .collect::<Vec<_>>()),
                q["html"],
                "{} HTML",
                c["name"]
            );
            assert_eq!(
                json!(tables
                    .iter()
                    .map(|t| spreadsheet::table::block(t, true))
                    .collect::<Vec<_>>()),
                q["blocks"],
                "{} blocks",
                c["name"]
            );
            if let Some(gap) = gap {
                let regions = spreadsheet::regions::discover(&sheet, gap);
                assert_eq!(
                    serde_json::to_value(&regions).unwrap(),
                    q["regions"],
                    "{} regions",
                    c["name"]
                );
                let summary = spreadsheet::regions::summarize(&sheet, &regions);
                let actual = serde_json::to_value(&summary).unwrap();
                for (k, v) in q["summary"].as_object().unwrap() {
                    assert!(
                        (actual[k].as_f64().unwrap() - v.as_f64().unwrap()).abs() < 1e-12,
                        "{} summary {k}",
                        c["name"]
                    );
                }
            }
        }
        if let Some(selection) = c["synthetic"].as_object() {
            let rows = serde_json::from_value::<Vec<u32>>(selection["rows"].clone()).unwrap();
            let cols = serde_json::from_value::<Vec<u32>>(selection["cols"].clone()).unwrap();
            let table = sheet.synthetic(&rows, &cols).unwrap();
            assert_eq!(
                serde_json::to_value(&table).unwrap(),
                selection["table"],
                "{} synthetic",
                c["name"]
            );
            assert_eq!(
                spreadsheet::table::html(&table),
                selection["html"],
                "{} synthetic HTML",
                c["name"]
            );
        }
    }
}
