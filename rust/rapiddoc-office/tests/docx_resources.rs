use rapiddoc_office::docx::{
    self,
    style_map::Rules,
    table_html::{serialize, Html},
};
use serde_json::Value;
use std::io::Read;
fn fixture() -> Value {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_resources.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    serde_json::from_slice(&raw).unwrap()
}
#[test]
fn original_embedded_style_map_grammar_and_matching() {
    let f = fixture();
    for case in f["grammar"].as_array().unwrap() {
        let rules = Rules::parse(case["text"].as_str().unwrap());
        assert_eq!(
            serde_json::json!(rules.warnings),
            case["warnings"],
            "{} warnings",
            case["text"]
        );
        for (q, expected) in f["queries"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["outputs"].as_array().unwrap())
        {
            let numbering = q["numbering"]
                .as_array()
                .map(|v| (v[0].as_str().unwrap(), v[1].as_bool().unwrap()));
            let result = rules
                .find(
                    q["kind"].as_str().unwrap(),
                    q["id"].as_str(),
                    q["name"].as_str(),
                    numbering,
                    q["qualifier"].as_str(),
                )
                .map(|p| {
                    let mut nodes = p.wrap(vec![Html::Text("A & <B>".into())]);
                    nodes.extend(p.wrap(vec![Html::Text("C".into())]));
                    serialize(nodes)
                });
            assert_eq!(serde_json::json!(result), *expected, "{} {q}", case["text"]);
        }
    }
}
#[test]
fn original_docx_notes_comments_style_maps_and_fallback() {
    let f = fixture();
    for case in f["cases"].as_array().unwrap() {
        let bytes = case["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let actual = docx::Source::open(bytes.into()).and_then(docx::body::convert);
        if case.get("error").is_some() {
            assert!(actual.is_err(), "{} expected error", case["name"]);
            continue;
        }
        let pages = serde_json::to_value(
            actual
                .unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
                .pages,
        )
        .unwrap();
        if pages != case["pages"] {
            let build = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build");
            std::fs::create_dir_all(&build).unwrap();
            std::fs::write(build.join("docx-resources-diff.json"),serde_json::to_vec_pretty(&serde_json::json!({"name":case["name"],"actual":pages,"expected":case["pages"]})).unwrap()).unwrap();
            panic!(
                "{} differs; see build/docx-resources-diff.json",
                case["name"]
            );
        }
    }
}
