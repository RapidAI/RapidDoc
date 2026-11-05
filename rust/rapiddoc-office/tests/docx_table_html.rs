use rapiddoc_office::{
    docx::{
        table_html::{normalize_colspans, serialize, Html, Reader, UnavailableResources},
        table_images::Images,
        Source,
    },
    package::Relationship,
    xml,
};
use serde_json::Value;
use std::io::Read;

fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/docx_table_html.json.gz"
        )[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn render(case: &Value, normalized: bool) -> rapiddoc_office::Result<String> {
    let tree = xml::parse(case["xml"].as_str().unwrap()).unwrap();
    if let Some(path) = case["source"].as_str() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut source = Source::open(std::fs::read(root.join(path)).unwrap().into()).unwrap();
        let images = Images::new(&mut source.package, &source.relationships).unwrap();
        let mut reader = Reader::new(
            case["styles"].as_str(),
            case["numbering"].as_str(),
            &source.relationships,
            images,
        )
        .unwrap();
        if normalized {
            reader.normalized_table_html(tree.root_element())
        } else {
            reader.table_html(tree.root_element())
        }
    } else {
        let relationships: Vec<_> = case["relationships"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Relationship {
                id: r["id"].as_str().unwrap().into(),
                target: r["target"].as_str().unwrap().into(),
                kind: r["kind"].as_str().unwrap().into(),
                external: true,
                resolved: None,
            })
            .collect();
        let mut reader = Reader::new(
            case["styles"].as_str(),
            case["numbering"].as_str(),
            &relationships,
            UnavailableResources,
        )
        .unwrap();
        if normalized {
            reader.normalized_table_html(tree.root_element())
        } else {
            reader.table_html(tree.root_element())
        }
    }
}
#[test]
fn original_mammoth_table_content_html() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = render(case, false);
        if !case["pending_resources"].as_array().unwrap().is_empty() {
            assert_eq!(
                actual.unwrap_err().kind,
                rapiddoc_office::ErrorKind::Unsupported,
                "{} must preserve the pending equation path",
                case["name"]
            );
            continue;
        }
        if case.get("error").is_some() {
            assert!(
                actual.is_err(),
                "{} expected {}: {actual:?}",
                case["name"],
                case["error"]
            );
        } else {
            assert_eq!(
                actual.unwrap_or_else(|e| panic!("{}: {e}", case["name"])),
                case["html"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}
fn canonical(html: &str) -> Value {
    let input = format!("<root>{html}</root>");
    let tree = xml::parse(&input).unwrap();
    fn node(n: roxmltree::Node<'_, '_>) -> Value {
        if n.is_text() {
            return Value::String(n.text().unwrap().into());
        }
        serde_json::json!({"tag":n.tag_name().name(),"attributes":n.attributes().map(|a| (a.name().to_owned(),a.value().to_owned())).collect::<std::collections::BTreeMap<_,_>>(),"children":n.children().filter(|n| n.is_text() || n.is_element()).map(node).collect::<Vec<_>>()})
    }
    node(tree.root_element())
}
#[test]
fn local_colspan_correction_preserves_content() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        if case.get("error").is_some() || !case["pending_resources"].as_array().unwrap().is_empty()
        {
            continue;
        }
        let actual = render(case, true).unwrap();
        assert_eq!(
            canonical(&actual),
            canonical(case["normalized_html"].as_str().unwrap()),
            "{}",
            case["name"]
        );
    }
}
#[test]
fn colspan_original_edge_cases_and_arbitrary_precision() {
    fn read(node: roxmltree::Node<'_, '_>) -> Html {
        if node.is_text() {
            return Html::Text(node.text().unwrap().into());
        }
        let mut out = Html::element(
            node.tag_name().name(),
            false,
            node.children()
                .filter(|n| n.is_text() || n.is_element())
                .map(read)
                .collect(),
        );
        for a in node.attributes() {
            out = out.attribute(a.name(), a.value());
        }
        out
    }
    for case in fixture()["normalizations"].as_array().unwrap() {
        let input = format!("<root>{}</root>", case["html"].as_str().unwrap());
        let tree = xml::parse(&input).unwrap();
        let mut nodes: Vec<_> = tree.root_element().children().map(read).collect();
        normalize_colspans(&mut nodes);
        assert_eq!(
            canonical(&serialize(nodes)),
            canonical(case["expected"].as_str().unwrap()),
            "{}",
            case["name"]
        );
    }
}
