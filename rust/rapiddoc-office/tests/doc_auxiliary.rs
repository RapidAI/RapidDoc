use rapiddoc_office::{
    doc::{auxiliary, body},
    docx::{
        body::{Content, Kind},
        spans::Span,
    },
    ole::OleReader,
};
fn flatten(spans: &[Span], url: Option<&str>) -> Vec<serde_json::Value> {
    let mut out = vec![];
    for s in spans {
        match s {
            Span::Text { content, styles } => {
                for c in content.chars() {
                    out.push(serde_json::json!({"text":c.to_string(),"styles":styles,"url":url}))
                }
            }
            Span::Link { url, content } => out.extend(flatten(content, Some(url))),
            Span::Equation { .. } => panic!("unexpected equation in text reference"),
        }
    }
    out
}
#[test]
fn real_headers_footers_match_python_content_styles_and_links() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let b = body::convert(&mut ole).unwrap();
    let actual:Vec<Vec<_>>=b.pages.iter().map(|p|p.iter().filter(|b|matches!(b.kind,Kind::Header|Kind::Footer)).map(|b|{let Content::Spans(s)=&b.content else{panic!()};serde_json::json!({"kind":if matches!(b.kind,Kind::Header){"header"}else{"footer"},"characters":flatten(s,None)})}).collect()).collect();
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_auxiliary.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(serde_json::json!(actual), expected);
}
#[test]
fn dedup_is_local_and_page_numbers_filtered() {
    let s = |text: &str| {
        vec![Span::Text {
            content: text.into(),
            styles: vec![],
        }]
    };
    assert_eq!(
        auxiliary::contents(vec![s("1"), s("footer"), s("footer")]).len(),
        1
    );
    assert_eq!(auxiliary::contents(vec![s("footer")]).len(), 1);
    assert_eq!(
        auxiliary::contents(vec![s("１２３"), s("²"), s("Ⅳ")]).len(),
        1
    );
}
