use rapiddoc_office::docx::{
    embedded::Shared,
    formulas::{self, BeforeInjection, Kind, Token, Tokens},
    table_html::Reader,
    Context, Source,
};
use serde_json::{json, Value};
use std::io::Read;
fn canonical(input: &str) -> Value {
    let input = format!("<root>{}</root>", input.replace("<br>", "<br/>"));
    let tree = rapiddoc_office::xml::parse(&input).unwrap();
    fn node(n: roxmltree::Node<'_, '_>) -> Value {
        if n.is_text() {
            return json!(n.text().unwrap());
        }
        json!({"tag":n.tag_name().name(),"attrs":n.attributes().map(|a| (a.name(),a.value())).collect::<std::collections::BTreeMap<_,_>>(),"children":n.children().filter(|n| n.is_text() || n.is_element()).map(node).collect::<Vec<_>>()})
    }
    node(tree.root_element())
}
#[test]
fn docx_per_part_binary_and_image_equations_preview_and_tables() {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_embedded.json.gz")
            [..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    let fixture: Value = serde_json::from_slice(&bytes).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let bytes: Vec<_> = case["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect();
        let mut source = Source::open(bytes.into()).unwrap();
        let context = Context::from_parts(
            &source.document,
            source.styles.as_deref(),
            source.numbering.as_deref(),
            &source.relationships,
        )
        .unwrap();
        let shared = Shared::new(&mut source.package, &source.relationships).unwrap();
        let mut tokens = Tokens::new(shared.embedded());
        let mut reader = Reader::new(
            source.styles.as_deref(),
            source.numbering.as_deref(),
            &source.relationships,
            BeforeInjection(shared.images()),
        )
        .unwrap();
        let tables: Vec<_> = context
            .tables()
            .filter(|n| n.parent().is_some_and(|p| p.tag_name().name() == "body"))
            .collect();
        assert_eq!(tables.len(), case["tables"].as_array().unwrap().len());
        for (table, expected) in tables.into_iter().zip(case["tables"].as_array().unwrap()) {
            let actual: Vec<_> = tokens
                .extract(table)
                .unwrap()
                .into_iter()
                .map(|t| match t {
                    Token::Text(s) => json!(["text", s]),
                    Token::Formula(k, s) => json!([
                        match k {
                            Kind::Omml => "omml",
                            Kind::EquationXml => "equationxml",
                            Kind::Mtef => "mtef",
                            Kind::ImageMtef => "image_mtef",
                        },
                        s
                    ]),
                })
                .collect();
            assert_eq!(
                json!(actual),
                expected["tokens"],
                "{} table {} tokens",
                case["name"],
                expected["index"]
            );
            let previews: Vec<_> = table
                .descendants()
                .filter(|n| n.is_element() && matches!(n.tag_name().name(), "blip" | "imagedata"))
                .map(|n| shared.is_equation_preview(n).unwrap())
                .collect();
            assert_eq!(
                json!(previews),
                expected["previews"],
                "{} previews",
                case["name"]
            );
            let html = formulas::table(&context, &mut reader, table, &mut tokens)
                .unwrap_or_else(|e| panic!("{} table {}: {e}", case["name"], expected["index"]));
            assert_eq!(
                canonical(&html),
                canonical(expected["safe_html"].as_str().unwrap()),
                "{} table {}",
                case["name"],
                expected["index"]
            );
        }
        assert_eq!(
            shared.candidate_total_bytes() as u64,
            case["image_bytes"].as_u64().unwrap(),
            "{} image bytes",
            case["name"]
        );
        assert_eq!(
            shared.ole_total_bytes() as u64,
            case["ole_bytes"].as_u64().unwrap(),
            "{} OLE bytes",
            case["name"]
        );
    }
}
