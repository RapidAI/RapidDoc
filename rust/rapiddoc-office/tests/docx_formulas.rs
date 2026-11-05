use rapiddoc_office::{
    docx::{
        formulas::{self, BeforeInjection, Kind, NativeEmbedded, Token, Tokens},
        table_html::{self, Reader, UnavailableResources},
        Context,
    },
    equation::equation_xml,
    links,
    package::Relationship,
    xml,
};
use serde_json::{json, Value};
use std::io::Read;
fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_formulas.json.gz")
            [..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn canonical(html: &str) -> Value {
    // Reference plain <br> is a normal HTML void element, not an XML open tag.
    let input = format!("<root>{}</root>", html.replace("<br>", "<br/>"));
    let tree = xml::parse(&input).unwrap();
    fn node(n: roxmltree::Node<'_, '_>) -> Value {
        if n.is_text() {
            return Value::String(n.text().unwrap().into());
        }
        json!({"tag":n.tag_name().name(),"attributes":n.attributes().map(|a| (a.name().to_owned(),a.value().to_owned())).collect::<std::collections::BTreeMap<_,_>>(),"children":n.children().filter(|n| n.is_text() || n.is_element()).map(node).collect::<Vec<_>>()})
    }
    node(tree.root_element())
}
#[test]
fn python_formula_tokens_and_word_cell_html() {
    for case in fixture()["cases"].as_array().unwrap() {
        let relationships: Vec<_> = case["relationships"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Relationship {
                id: r["id"].as_str().unwrap().into(),
                kind: r["kind"].as_str().unwrap().into(),
                target: r["target"].as_str().unwrap().into(),
                external: true,
                resolved: None,
            })
            .collect();
        let context = Context::from_parts(
            case["document"].as_str().unwrap(),
            case["styles"].as_str(),
            case["numbering"].as_str(),
            &relationships,
        )
        .unwrap();
        let table = case.get("table").is_some();
        let source = if table {
            context.tables().next().unwrap()
        } else {
            context.paragraphs().next().unwrap()
        };
        let mut tokens = Tokens::new(NativeEmbedded::default());
        let actual: Vec<_> = tokens
            .extract(source)
            .unwrap()
            .into_iter()
            .map(|token| match token {
                Token::Text(text) => json!(["text", text]),
                Token::Formula(kind, text) => json!([
                    match kind {
                        Kind::Omml => "omml",
                        Kind::EquationXml => "equationxml",
                        Kind::Mtef => "mtef",
                        Kind::ImageMtef => "image_mtef",
                    },
                    text
                ]),
            })
            .collect();
        assert_eq!(json!(actual), case["tokens"], "{} tokens", case["name"]);
        if table {
            let mut reader = Reader::new(
                case["styles"].as_str(),
                case["numbering"].as_str(),
                &relationships,
                BeforeInjection(UnavailableResources),
            )
            .unwrap();
            let output = formulas::table(&context, &mut reader, source, &mut tokens).unwrap();
            assert_eq!(
                canonical(&output),
                canonical(case["safe_table"].as_str().unwrap()),
                "{} table",
                case["name"]
            );
        } else {
            let output = formulas::paragraph(&context, source, &mut tokens)
                .unwrap()
                .map(|p| table_html::serialize(vec![p]));
            if case["safe_paragraph"].is_null() {
                assert!(output.is_none(), "{}", case["name"]);
            } else {
                assert_eq!(
                    canonical(output.as_deref().unwrap()),
                    canonical(case["safe_paragraph"].as_str().unwrap()),
                    "{} paragraph",
                    case["name"]
                );
            }
        }
    }
}
#[test]
fn original_equation_xml_structure_and_cache_accounting() {
    for case in fixture()["equationxml"].as_array().unwrap() {
        let mut decoder = equation_xml::Decoder::default();
        let actual = decoder.decode(case["input"].as_str()).unwrap();
        assert_eq!(json!(actual), case["expected"], "{}", case["name"]);
        assert_eq!(decoder.decode(case["input"].as_str()).unwrap(), actual);
        assert_eq!(
            decoder.total_bytes as u64,
            case["total_bytes"].as_u64().unwrap(),
            "{}",
            case["name"]
        );
    }
}
#[test]
fn original_office_link_policy() {
    for case in fixture()["links"].as_array().unwrap() {
        assert_eq!(
            json!(links::sanitize(case["input"].as_str())),
            case["expected"],
            "{}",
            case["input"]
        );
    }
}
