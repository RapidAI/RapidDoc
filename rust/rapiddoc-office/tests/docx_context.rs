use rapiddoc_office::{
    docx::{Context, Source},
    package::Relationship,
};
use serde_json::{json, Value};
use std::{io::Read, path::PathBuf};

fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_context.json.gz")[..])
        .read_to_end(&mut bytes).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn relationships(case: &Value) -> Vec<Relationship> {
    case["relationships"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| Relationship {
            id: r["id"].as_str().unwrap().to_owned(),
            kind: r["kind"].as_str().unwrap().to_owned(),
            target: r["target"].as_str().unwrap().to_owned(),
            external: r["external"].as_bool().unwrap(),
            resolved: None,
        })
        .collect()
}
fn compare(context: &Context<'_>, case: &Value) {
    let paragraphs = context.paragraphs().collect::<Vec<_>>();
    assert_eq!(
        paragraphs.len(),
        case["paragraphs"].as_array().unwrap().len(),
        "{}",
        case["name"]
    );
    for (index, (paragraph, expected)) in paragraphs
        .into_iter()
        .zip(case["paragraphs"].as_array().unwrap())
        .enumerate()
    {
        let actual = context.paragraph_facts(paragraph);
        if expected.get("error").is_some() {
            assert!(
                actual.is_err(),
                "{} paragraph {index}: {actual:?}",
                case["name"]
            );
        } else {
            assert_eq!(
                serde_json::to_value(
                    actual.unwrap_or_else(|e| panic!("{} paragraph {index}: {e}", case["name"]))
                )
                .unwrap(),
                *expected,
                "{} paragraph {index}",
                case["name"]
            );
        }
    }
}
#[test]
fn python_docx_paragraph_contexts() {
    for case in fixture()["cases"].as_array().unwrap() {
        let context = Context::from_parts(
            case["document"].as_str().unwrap(),
            case["styles"].as_str(),
            case["numbering"].as_str(),
            &relationships(case),
        )
        .unwrap();
        compare(&context, case);
    }
}
#[test]
fn real_docx_packages_use_original_parts_and_strict_namespaces() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for case in fixture()["cases"].as_array().unwrap() {
        let Some(path) = case["source"].as_str() else {
            continue;
        };
        let source = Source::open(std::fs::read(root.join(path)).unwrap().into()).unwrap();
        let context = source.context().unwrap();
        compare(&context, case);
    }
}
#[test]
fn python_numbering_counter_override_and_parent_restart() {
    for case in fixture()["cases"].as_array().unwrap() {
        let mut context = Context::from_parts(
            case["document"].as_str().unwrap(),
            case["styles"].as_str(),
            case["numbering"].as_str(),
            &relationships(case),
        )
        .unwrap();
        for expected in case["counters"].as_array().unwrap() {
            let id = expected["id"].as_i64().unwrap();
            let level = expected["level"].as_i64().unwrap();
            let start = context.numbering.start(id, level).map(|v| v.to_string());
            let numbered = context.numbering.is_numbered(id, level);
            let actual = context.numbering.advance(id, level);
            if expected.get("error").is_some() {
                assert!(actual.is_err());
            } else {
                assert_eq!(
                    json!({"id":id,"level":level,"start":start,"numbered":numbered,"next":actual.unwrap().map(|v|v.to_string())}),
                    *expected,
                    "{}",
                    case["name"]
                );
            }
        }
    }
}

#[test]
fn main_at_package_root_loads_relative_style_relationships() {
    use std::io::{Cursor, Write};
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let members = [
        (
            "_rels/.rels",
            r#"<Relationships><Relationship Id="main" Type="a/officeDocument" Target="main.xml"/></Relationships>"#,
        ),
        (
            "_rels/main.xml.rels",
            r#"<Relationships><Relationship Id="styles" Type="a/styles" Target="styles.xml"/></Relationships>"#,
        ),
        (
            "main.xml",
            r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Custom"/></w:pPr><w:r><w:t>Root document</w:t></w:r></w:p></w:body></w:document>"#,
        ),
        (
            "styles.xml",
            r#"<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Custom"><w:name w:val="Custom"/><w:rPr><w:b/></w:rPr></w:style></w:styles>"#,
        ),
    ];
    for (name, text) in members {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(text.as_bytes()).unwrap();
    }
    let source = Source::open(zip.finish().unwrap().into_inner().into()).unwrap();
    let context = source.context().unwrap();
    let paragraph = context.paragraphs().next().unwrap();
    let facts = context.paragraph_facts(paragraph).unwrap();
    assert_eq!(facts.label, "Custom");
    assert!(facts.elements[0].format.as_ref().unwrap().bold);
}
