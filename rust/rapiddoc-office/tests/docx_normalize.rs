use rapiddoc_office::{docx::normalize, package::Package};
use serde_json::{json, Value};
use std::io::Read;
#[test]
fn original_docx_reachable_graph_crc_and_missing_relationships() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_normalize.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let mut p = Package::open(data.into()).unwrap();
        let result = normalize::apply(&mut p);
        if c.get("error").is_some() {
            assert!(result.is_err(), "{} must fail", c["name"]);
            continue;
        }
        result.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let mut names = p.names().to_vec();
        names.sort();
        assert_eq!(json!(names), c["members"], "{}", c["name"]);
        assert_eq!(
            serde_json::to_value(p.content_types().unwrap().overrides).unwrap(),
            c["overrides"],
            "{} content types",
            c["name"]
        );
        for (name, ids) in c["rels"].as_object().unwrap() {
            let text = p.read_xml(name).unwrap().unwrap();
            let tree = rapiddoc_office::xml::parse(&text).unwrap();
            let actual = tree
                .root_element()
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
                .map(|n| n.attribute("Id").unwrap_or(""))
                .collect::<Vec<_>>();
            assert_eq!(json!(actual), *ids, "{} {name}", c["name"]);
        }
    }
}
