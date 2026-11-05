use rapiddoc_office::{
    doc::{
        body,
        fib::Pair,
        navigation,
        tables::{self, Item},
    },
    docx::{
        body::{Content, Kind},
        spans::Span,
    },
    ole::OleReader,
};
use serde_json::{json, Value};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture(name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(
            root()
                .join("tests/fixtures/office_rust_reference_068")
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}
fn characters(spans: &Value, url: Option<&str>) -> Vec<Value> {
    let mut out = vec![];
    for s in spans.as_array().unwrap() {
        if s["type"] == "hyperlink" {
            out.extend(characters(&s["content"], s["url"].as_str()));
        } else {
            for c in s["content"].as_str().unwrap().chars() {
                out.push(json!({"text":c.to_string(),"kind":s["type"],"styles":s.get("styles").cloned().unwrap_or(json!([])),"url":url}));
            }
        }
    }
    out
}
fn nav(b: &Value) -> Value {
    let mut out = json!({"kind":b["type"]});
    for k in [
        "anchor",
        "list_label",
        "ilevel",
        "attribute",
        "start",
        "level",
        "is_numbered_style",
    ] {
        out[k] = b[k].clone();
    }
    if b["type"] == "list" || b["type"] == "index" {
        out["children"] = b["content"].as_array().unwrap().iter().map(nav).collect();
    } else {
        out["characters"] = json!(characters(&b["content"], None));
    }
    out
}
#[test]
fn real_navigation_and_table_content_match_python() {
    let mut ole = OleReader::open(
        std::fs::read(root().join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let body = body::convert(&mut ole).unwrap();
    let raw = serde_json::to_value(&body).unwrap();
    std::fs::write(
        root().join("build/doc-navigation-raw.json"),
        serde_json::to_vec(&raw).unwrap(),
    )
    .unwrap();
    let pages: Vec<Vec<_>> = raw["pages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p.as_array()
                .unwrap()
                .iter()
                .filter(|b| {
                    matches!(
                        b["type"].as_str(),
                        Some("doc_title" | "paragraph_title" | "list" | "index")
                    )
                })
                .map(nav)
                .collect()
        })
        .collect();
    assert_eq!(json!(pages), fixture("doc_navigation_real.json"));
    let image = regex::Regex::new(r#"src="[^"]*""#).unwrap();
    let tables: Vec<_> = raw["pages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|p| p.as_array().unwrap())
        .filter(|b| b["type"] == "table")
        .map(|b| {
            image
                .replace_all(b["content"].as_str().unwrap(), "src=\"IMAGE\"")
                .into_owned()
        })
        .collect();
    assert_eq!(json!(tables), fixture("doc_tables_real.json"));
    let mut output = rapiddoc_office::document_output::middle(body.pages, "test").unwrap();
    for page in &mut output.middle.pdf_info {
        page.para_blocks
            .retain(|b| matches!(b.kind.as_str(), "title" | "list" | "index"));
        page.discarded_blocks.clear();
    }
    fn clean(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.remove("index");
                for v in m.values_mut() {
                    clean(v)
                }
            }
            Value::Array(v) => {
                for v in v {
                    clean(v)
                }
            }
            _ => {}
        }
    }
    let mut public = json!(output
        .middle
        .pdf_info
        .iter()
        .map(|p| &p.para_blocks)
        .collect::<Vec<_>>());
    clean(&mut public);
    assert_eq!(public, fixture("doc_navigation_public.json"));
    let exports = fixture("doc_navigation_exports.json");
    for mode in [
        "mm_markdown",
        "nlp_markdown",
        "content_list",
        "content_list_v2",
    ] {
        assert_eq!(
            rapiddoc_office::document_output::render::export(
                &output.middle,
                mode,
                "images",
                &Default::default()
            ),
            exports[mode],
            "{mode}"
        );
    }
}
#[test]
fn binary_bookmarks_match_python() {
    for c in fixture("doc_bookmarks.json").as_array().unwrap() {
        let data: Vec<u8> = c["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let n = c["sizes"][0].as_u64().unwrap() as u32;
        let s = c["sizes"][1].as_u64().unwrap() as u32;
        let e = c["sizes"][2].as_u64().unwrap() as u32;
        let b = navigation::bookmarks(
            &data,
            Pair { fc: 0, lcb: n },
            Pair { fc: n, lcb: s },
            Pair { fc: n + s, lcb: e },
        );
        assert_eq!(json!(b), c["expected"]);
        assert_eq!(navigation::anchor(&b, 0, 4), Some("_Toc1".into()));
    }
}
#[test]
fn toc_page_suffixes_match_python_without_changing_numbers_in_titles() {
    for c in fixture("doc_toc_runs.json").as_array().unwrap() {
        let runs = c["runs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let text = Span::Text {
                    content: v[0].as_str().unwrap().into(),
                    styles: vec![],
                };
                if let Some(url) = v[1].as_str() {
                    Span::Link {
                        url: url.into(),
                        content: vec![text],
                    }
                } else {
                    text
                }
            })
            .collect();
        let actual = serde_json::to_value(navigation::toc_runs(runs)).unwrap();
        assert_eq!(json!(characters(&actual, None)), c["characters"], "{c}");
    }
}
#[test]
fn nested_cell_lists_and_type_boundaries_match_python() {
    for c in fixture("doc_cell_lists.json").as_array().unwrap() {
        let mut ole = OleReader::open(
            std::fs::read(root().join("tests/migration/inputs/synthetic.doc"))
                .unwrap()
                .into(),
        )
        .unwrap();
        let _ = body::convert(&mut ole).unwrap();
        let count = c["paragraphs"].as_array().unwrap().len();
        let items = c["paragraphs"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut ole = OleReader::open(
                    std::fs::read(root().join("tests/migration/inputs/synthetic.doc"))
                        .unwrap()
                        .into(),
                )
                .unwrap();
                let mut block = body::convert(&mut ole).unwrap().pages[0].remove(0);
                block.kind = Kind::Text;
                block.content = Content::None;
                let text = tables::escape(p["text"].as_str().unwrap(), false);
                let cell_list = (!p["list"].is_null()).then(|| {
                    let l = &p["list"];
                    let ordered = l["ordered"].as_bool().unwrap();
                    let label = if ordered {
                        format!("{} ", tables::escape(l["label"].as_str().unwrap(), true))
                    } else {
                        String::new()
                    };
                    (
                        ordered,
                        l["level"].as_u64().unwrap() as u8,
                        format!("<li>{label}{text}</li>"),
                    )
                });
                Item {
                    cp: i as u32,
                    pap: rapiddoc_office::doc::paragraph::Paragraph {
                        in_table: Some(true),
                        table_depth: Some(1),
                        ..Default::default()
                    },
                    cell: i + 1 == count,
                    row: false,
                    visible: true,
                    html: format!("<p>{text}</p>"),
                    block,
                    visuals: vec![],
                    list_id: None,
                    cell_list,
                }
            })
            .collect();
        let result = tables::assemble(items);
        let Content::Text(html) = &result[0].1.content else {
            panic!()
        };
        assert_eq!(html, c["html"].as_str().unwrap());
    }
}
