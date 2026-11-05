use rapiddoc_office::{
    doc::{body, fields::Fields, resources::Visual},
    docx::spans::{self, Span},
    ole::OleReader,
    xls::embedded_chart,
};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn instruction(fields: &mut Fields, out: &mut Vec<Span>, text: &str) {
    fields.begin();
    fields.push(
        out,
        Span::Text {
            content: text.into(),
            styles: vec![],
        },
    );
    fields.separator();
}
#[test]
fn native_formula_suppresses_preview_and_survives_parent_link() {
    let mut fields = Fields::default();
    let mut out = vec![];
    instruction(&mut fields, &mut out, "HYPERLINK https://example.com");
    instruction(&mut fields, &mut out, "EMBED Equation.3");
    fields.bind_object(&mut out, Some("x^2"), None);
    assert!(fields.native_formula());
    fields.end(&mut out);
    fields.end(&mut out);
    assert!(
        matches!(&out[0],Span::Link{content,..} if matches!(&content[0],Span::Equation{content} if content=="x^2"))
    );
}
#[test]
fn charts_bind_only_to_chart_fields_and_emit_once_with_or_without_preview() {
    for command in ["EMBED Excel.Chart.8", "\"EMBED\" \"MSGraph.Chart.8\""] {
        let mut fields = Fields::default();
        let mut out = vec![];
        instruction(&mut fields, &mut out, command);
        fields.bind_object(&mut out, None, Some("<table/>"));
        let chart = fields
            .chart_picture(Some("data:image/png;base64,AA==".into()))
            .unwrap();
        assert!(matches!(
            chart,
            Visual::Chart {
                preview: Some(_),
                ..
            }
        ));
        assert!(fields.end_with_visual(&mut out).is_none());
        instruction(&mut fields, &mut out, command);
        fields.bind_object(&mut out, None, Some("<table/>"));
        assert!(matches!(
            fields.end_with_visual(&mut out),
            Some(Visual::Chart { preview: None, .. })
        ));
    }
    let mut fields = Fields::default();
    let mut out = vec![];
    instruction(&mut fields, &mut out, "EMBED Excel.Sheet.8");
    fields.bind_object(&mut out, None, Some("wrong"));
    fields.push(
        &mut out,
        Span::Text {
            content: "cached".into(),
            styles: vec![],
        },
    );
    assert!(fields.chart_picture(None).is_none());
    fields.end(&mut out);
    assert_eq!(spans::visible(&out), "cached");
}
#[test]
fn workbook_and_graph_html_match_python() {
    let path = root().join("tests/fixtures/office_rust_reference_068");
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(path.join("doc_embedded_charts.json")).unwrap())
            .unwrap();
    for c in cases {
        let bytes = std::fs::read(path.join(c["file"].as_str().unwrap())).unwrap();
        assert_eq!(
            serde_json::to_value(embedded_chart::extract(bytes.into()).unwrap()).unwrap(),
            c["html"],
            "{}",
            c["file"]
        );
    }
}
#[test]
fn real_visual_order_and_object_content_match_python() {
    let mut ole = OleReader::open(
        std::fs::read(root().join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let body = body::convert(&mut ole).unwrap();
    let raw = serde_json::to_value(body).unwrap();
    fn collect(b: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
        if matches!(b["type"].as_str(), Some("image" | "equation" | "chart")) {
            out.push(serde_json::json!({"type":b["type"],"content":if b["type"]=="image"{serde_json::Value::Null}else{b["content"].clone()}}));
        } else if b["type"] == "list" {
            for child in b["content"].as_array().unwrap() {
                collect(child, out)
            }
        }
    }
    let actual: Vec<Vec<_>> = raw["pages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|page| {
            let mut out = vec![];
            for b in page.as_array().unwrap() {
                collect(b, &mut out)
            }
            out
        })
        .collect();
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/office_rust_reference_068/doc_visuals.json"))
            .unwrap(),
    )
    .unwrap();
    std::fs::write(
        root().join("build/doc-objects-raw.json"),
        serde_json::to_vec(&raw).unwrap(),
    )
    .unwrap();
    assert_eq!(serde_json::json!(actual), expected);
}
