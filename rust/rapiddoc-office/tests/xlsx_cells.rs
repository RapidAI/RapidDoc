use rapiddoc_office::{spreadsheet, xlsx};
use serde_json::{json, Value};
use std::io::Read;
#[test]
fn original_xlsx_package_cells_styles_and_table_projection() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/xlsx_cells.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = c["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|s| u8::from_str_radix(std::str::from_utf8(s).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let bytes: std::sync::Arc<[u8]> = bytes.into();
        let actual = xlsx::Workbook::open(bytes.clone());
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected error", c["name"]);
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        assert_eq!(
            actual.date1904,
            c["date1904"].as_bool().unwrap(),
            "{} epoch",
            c["name"]
        );
        assert_eq!(
            actual.sheets.len(),
            c["sheets"].as_array().unwrap().len(),
            "{} sheets",
            c["name"]
        );
        for (sheet, expected) in actual.sheets.iter().zip(c["sheets"].as_array().unwrap()) {
            assert_eq!(sheet.name, expected["name"], "{} title", c["name"]);
            assert_eq!(sheet.state, expected["state"], "{} state", c["name"]);
            let cells=sheet.cells.cells.values().map(|v|json!({"row":v.row,"col":v.col,"has_value":v.has_value,"semantic":v.semantic,"cell":v.cell})).collect::<Vec<_>>();
            if json!(cells) != expected["cells"] {
                let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/xlsx-cell-diff.json");
                std::fs::write(p,serde_json::to_vec_pretty(&json!({"name":c["name"],"sheet":sheet.name,"actual":cells,"expected":expected["cells"]})).unwrap()).unwrap();
                panic!(
                    "{} {} cells differ; build/xlsx-cell-diff.json",
                    c["name"], sheet.name
                );
            }
            assert_eq!(
                json!(sheet.cells.default_cell),
                expected["default_cell"],
                "{} {} default cell",
                c["name"],
                sheet.name
            );
            let (gap, score, tables) = sheet.cells.project(None);
            assert_eq!(json!(gap), expected["selected_gap"], "{} gap", c["name"]);
            assert!(
                (score - expected["penalty"].as_f64().unwrap()).abs() < 1e-12,
                "{} score",
                c["name"]
            );
            assert_eq!(
                json!(tables),
                expected["tables"],
                "{} {} tables",
                c["name"],
                sheet.name
            );
            assert_eq!(
                json!(tables
                    .iter()
                    .map(spreadsheet::table::html)
                    .collect::<Vec<_>>()),
                expected["html"],
                "{} {} HTML",
                c["name"],
                sheet.name
            );
        }
        if let Some(queries) = c["full_cells_queries"].as_array() {
            for q in queries {
                let complete = xlsx::Workbook::open(bytes.clone())
                    .unwrap()
                    .into_pages(
                        q["include_hidden"].as_bool().unwrap(),
                        q["gap"].as_u64().map(|v| v as u32),
                        q["singleton"].as_bool().unwrap(),
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "{} resource-capable conversion of cell-only input: {e}",
                            c["name"]
                        )
                    });
                assert_eq!(
                    json!(complete),
                    q["pages"],
                    "{} resource-capable cell-only output",
                    c["name"]
                );
                let pages = actual
                    .convert_cells(
                        q["include_hidden"].as_bool().unwrap(),
                        q["gap"].as_u64().map(|v| v as u32),
                        q["singleton"].as_bool().unwrap(),
                    )
                    .unwrap_or_else(|e| panic!("{} cell-only full conversion: {e}", c["name"]));
                assert_eq!(
                    json!(pages),
                    q["pages"],
                    "{} full file-to-pages {q}",
                    c["name"]
                );
            }
        }
    }
}
