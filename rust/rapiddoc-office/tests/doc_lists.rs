use rapiddoc_office::{
    doc::{body, fib, lists::ListTables},
    ole::OleReader,
};
#[test]
fn numbering_sequences_match_python() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let word = ole.read_stream("WordDocument", true).unwrap();
    let f = fib::parse(&word).unwrap();
    let t = ole.read_stream("1Table", true).unwrap();
    let mut lists = ListTables::load(&t, f.pair(73), f.pair(74));
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_lists.json"))
            .unwrap(),
    )
    .unwrap();
    for c in cases.as_array().unwrap() {
        let actual = serde_json::to_value(lists.next(
            c["id"].as_u64().unwrap() as u32,
            c["level"].as_u64().unwrap() as u8,
        ))
        .unwrap();
        assert_eq!(actual, c["expected"], "{c}");
    }
}
#[test]
fn inspect_real_doc_structure() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let body = body::convert(&mut ole).unwrap();
    let raw = serde_json::to_value(&body).unwrap();
    fn count(v: &serde_json::Value, k: &str) -> usize {
        match v {
            serde_json::Value::Object(m) => {
                usize::from(m.get("type").and_then(|v| v.as_str()) == Some(k))
                    + m.values().map(|v| count(v, k)).sum::<usize>()
            }
            serde_json::Value::Array(v) => v.iter().map(|v| count(v, k)).sum(),
            _ => 0,
        }
    }
    println!(
        "lists={} tables={} pages={}",
        count(&raw, "list"),
        count(&raw, "table"),
        body.pages.len()
    );
    assert_eq!(body.pages.len(), 3);
    fn tree(b: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({"attribute":b["attribute"],"ilevel":b["ilevel"],"start":b["start"],"children":b["content"].as_array().unwrap().iter().map(|v|if v["type"]=="list"{tree(v)}else{serde_json::json!("text")}).collect::<Vec<_>>()})
    }
    let actual: Vec<Vec<_>> = raw["pages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|page| {
            page.as_array()
                .unwrap()
                .iter()
                .filter(|b| b["type"] == "list")
                .map(tree)
                .collect()
        })
        .collect();
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_list_tree.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(serde_json::json!(actual), expected);
    std::fs::write(
        root.join("build/doc-refactor-raw.json"),
        serde_json::to_vec_pretty(&raw).unwrap(),
    )
    .unwrap();
}
