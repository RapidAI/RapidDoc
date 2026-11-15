use rapiddoc_inference::table_transformer as t;
use serde_json::{json, Value};

#[test]
#[ignore = "Generate build/migration/table-transformer-sync.json with export_table_transformer_sync.py"]
fn python_transformer_replay() {
    let _limits = rapiddoc_core::resources::python_compatible();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/table-transformer-sync.json");
    let data: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut failures = 0;
    for item in data.as_array().unwrap() {
        let objects = serde_json::from_value::<Vec<t::Object>>(item["objects"].clone()).unwrap();
        let tokens = serde_json::from_value::<Vec<t::Token>>(item["tokens"].clone()).unwrap();
        let cells = t::decode_objects(&objects, &tokens);
        let actual = json!(cells.iter().map(|c|json!({
            "bbox":c.bbox,"row_nums":c.row_nums,"column_nums":c.column_nums,
            "column_header":c.column_header,"text":c.text,
            "token_indices":c.token_indices().iter().map(|&i|tokens[i].line_num).collect::<Vec<_>>()
        })).collect::<Vec<_>>());
        if actual != item["cells"] {
            failures += 1;
            eprintln!(
                "{} differs: {} vs {} cells",
                item["id"],
                cells.len(),
                item["cells"].as_array().unwrap().len()
            );
            let target =
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
            std::fs::write(
                target.join(format!(
                    "transformer-{}-actual.json",
                    item["id"].as_str().unwrap()
                )),
                actual.to_string(),
            )
            .unwrap();
        }
    }
    println!(
        "{} cases, {failures} discrepancies",
        data.as_array().unwrap().len()
    );
    assert_eq!(failures, 0);
}
