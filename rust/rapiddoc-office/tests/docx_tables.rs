use rapiddoc_office::{docx::tables, xml};
use serde_json::{json, Value};
use std::io::Read;
fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_tables.json.gz")[..]).read_to_end(&mut bytes).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn signature(value: &Value) -> tables::Signature {
    tables::Signature {
        row_count: value["row_count"].as_u64().unwrap() as usize,
        cell_count: value["cell_count"].as_u64().unwrap() as usize,
        image_count: value["image_count"].as_u64().unwrap() as usize,
        text: value["text"].as_str().unwrap().to_owned(),
    }
}
#[test]
fn table_signatures_and_symbols_match_python() {
    let fixture = fixture();
    for case in fixture["tables"].as_array().unwrap() {
        let doc = xml::parse(case["xml"].as_str().unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(tables::signature(doc.root_element())).unwrap(),
            case["signature"],
            "{}",
            case["name"]
        );
    }
    for case in fixture["fragments"].as_array().unwrap() {
        let doc = xml::parse(case["xml"].as_str().unwrap()).unwrap();
        assert_eq!(
            tables::char_fragment(doc.root_element()),
            case["expected"].as_str().unwrap()
        );
    }
    for case in fixture["matches"].as_array().unwrap() {
        assert_eq!(
            tables::signatures_match(&signature(&case["xml"]), &signature(&case["html"])),
            case["expected"].as_bool().unwrap()
        );
    }
}
#[test]
fn merged_table_layout_matches_original_mammoth() {
    for case in fixture()["tables"].as_array().unwrap() {
        let text = case["xml"].as_str().unwrap();
        let doc = xml::parse(text).unwrap();
        let layout = tables::layout(doc.root_element());
        if case["layout"].get("error").is_some() {
            assert!(layout.is_err(), "{}", case["name"]);
            continue;
        }
        let layout = layout.unwrap();
        let rows=layout.rows.iter().map(|row|json!({"header":row.header,"cells":row.cells.iter().map(|cell|json!({"colspan":cell.colspan,"rowspan":cell.rowspan})).collect::<Vec<_>>()})).collect::<Vec<_>>();
        assert_eq!(
            json!({"rows":rows,"merge_disabled":layout.merge_disabled}),
            case["layout"],
            "{}",
            case["name"]
        );
        for row in layout.rows {
            for cell in row.cells {
                assert!(text[cell.source].starts_with("<w:tc"));
            }
        }
    }
}
