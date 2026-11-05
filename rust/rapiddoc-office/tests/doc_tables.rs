use rapiddoc_office::{
    doc::{
        paragraph::Paragraph,
        tables::{self, Item},
    },
    docx::body::{Block, Content},
};
#[test]
fn original_cell_html_style_combinations() {
    use rapiddoc_office::docx::spans::Span;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let refs: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_cell_html.json"))
            .unwrap(),
    )
    .unwrap();
    for c in refs.as_array().unwrap() {
        let styles = c["styles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| match s.as_str().unwrap() {
                "bold" => "bold",
                "italic" => "italic",
                "underline" => "underline",
                "emphasis" => "emphasis",
                "strikethrough" => "strikethrough",
                "superscript" => "superscript",
                _ => "subscript",
            })
            .collect();
        let span = Span::Text {
            content: c["text"].as_str().unwrap().into(),
            styles,
        };
        let spans = if let Some(url) = c["url"].as_str() {
            vec![Span::Link {
                url: url.into(),
                content: vec![span],
            }]
        } else {
            vec![span]
        };
        assert_eq!(tables::inline(&spans), c["html"].as_str().unwrap(), "{c}");
    }
}
#[test]
fn python_nested_and_implicit_outer_tables() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let refs: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_tables.json"))
            .unwrap(),
    )
    .unwrap();
    // Obtain a text block through the real parser, avoiding a public constructor.
    for c in refs.as_array().unwrap() {
        let mut items = vec![];
        for (i, d) in c["depths"].as_array().unwrap().iter().enumerate() {
            let mut ole = rapiddoc_office::ole::OleReader::open(
                std::fs::read(root.join("tests/migration/inputs/synthetic.doc"))
                    .unwrap()
                    .into(),
            )
            .unwrap();
            let mut body = rapiddoc_office::doc::body::convert(&mut ole).unwrap();
            let block: Block = body.pages[0].remove(0);
            items.push(Item {
                list_id: None,
                cell_list: None,
                visuals: vec![],
                cp: i as u32,
                pap: Paragraph {
                    in_table: Some(true),
                    table_depth: d.as_i64(),
                    ..Default::default()
                },
                cell: true,
                row: i % 2 == 1,
                visible: true,
                html: format!("<p>{i}</p>"),
                block,
            });
        }
        let result = tables::assemble(items);
        assert_eq!(result.len(), 1);
        let Content::Text(html) = &result[0].1.content else {
            panic!()
        };
        assert_eq!(html, c["html"].as_str().unwrap());
    }
}
