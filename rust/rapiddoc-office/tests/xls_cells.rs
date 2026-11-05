use rapiddoc_office::xls::Workbook;
use serde_json::{json, Value};
use std::io::Read;
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_biff_workbook_cells_recovery_and_projection() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/xls_cells.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let actual = Workbook::open(bytes(c["package_hex"].as_str().unwrap()).into());
        if c.get("error").is_some() && c.get("sheets").is_none() {
            assert!(actual.is_err(), "{} expected {}", c["name"], c["error"]);
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        assert_eq!(
            json!(actual.active_sheet_index),
            c["active_sheet_index"],
            "{} active sheet",
            c["name"]
        );
        assert_eq!(
            json!(actual.chart_sheets),
            c["chart_sheets"],
            "{} chart sheets",
            c["name"]
        );
        let sheets=actual.sheets.iter().map(|s|json!({"name":s.name,"visible":s.visible,"order":s.order,"recovered":s.recovered,"merges":s.merges,"cells":s.cells.values().collect::<Vec<_>>()})).collect::<Vec<_>>();
        if json!(sheets) != c["sheets"] {
            std::fs::write(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/xls-cells-diff.json"),
                serde_json::to_vec_pretty(&json!({"case":c,"actual":sheets})).unwrap(),
            )
            .unwrap();
            panic!("{} cells differ; build/xls-cells-diff.json", c["name"]);
        }
        if let Some(pages) = c.get("pages") {
            assert_eq!(
                json!(actual
                    .cell_pages()
                    .unwrap_or_else(|e| panic!("{} pages: {e}", c["name"]))),
                *pages,
                "{} pages",
                c["name"]
            );
        } else if c.get("error").is_some() {
            assert!(
                actual.cell_pages().is_err(),
                "{} projection error",
                c["name"]
            );
        } else if c["has_resources"] == true {
            assert!(actual.cell_pages().is_err(), "{} resource guard", c["name"]);
        }
        if let Some(projected) = c["projected"].as_array() {
            for (sheet, expected) in actual.sheets.iter().zip(projected) {
                let native = sheet.project_cells().unwrap();
                for reference in expected["cells"].as_array().unwrap() {
                    let p = (
                        reference["row"].as_u64().unwrap() as u32,
                        reference["col"].as_u64().unwrap() as u32,
                    );
                    if sheet.merges.iter().any(|&(a, b, c, d)| {
                        p.0 >= a && p.0 <= c && p.1 >= b && p.1 <= d && p != (a, b)
                    }) {
                        continue;
                    }
                    let Some(cell) = native.cells.get(&p) else {
                        assert_eq!(reference["has_value"], false, "{} missing {p:?}", c["name"]);
                        continue;
                    };
                    assert_eq!(
                        json!({"row":cell.row,"col":cell.col,"has_value":cell.has_value,"semantic":cell.semantic,"cell":cell.cell}),
                        *reference,
                        "{} source cell {p:?}",
                        c["name"]
                    );
                }
            }
        }
    }
}
