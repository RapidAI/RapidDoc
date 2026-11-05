use rapiddoc_office::docx::{
    formulas::{Kind, Token},
    spans, Element,
};
use serde_json::Value;
use std::io::Read;
#[test]
fn original_rich_spans_links_unicode_slicing_and_formulas() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_spans.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        let elements = c["elements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| Element {
                text: e["text"].as_str().unwrap().into(),
                format: if e["format"].is_null() {
                    None
                } else {
                    let f = &e["format"];
                    Some(rapiddoc_office::docx::Formatting {
                        bold: f["bold"].as_bool().unwrap(),
                        italic: f["italic"].as_bool().unwrap(),
                        underline: f["underline"].as_bool().unwrap(),
                        underline_style: f["underline_style"].as_str().unwrap().into(),
                        emphasis: f["emphasis"].as_bool().unwrap(),
                        strikethrough: f["strikethrough"].as_bool().unwrap(),
                        script: match f["script"].as_str().unwrap() {
                            "super" => rapiddoc_office::docx::Script::Super,
                            "sub" => rapiddoc_office::docx::Script::Sub,
                            _ => rapiddoc_office::docx::Script::Baseline,
                        },
                    })
                },
                hyperlink: e["hyperlink"].as_str().map(str::to_owned),
            })
            .collect::<Vec<_>>();
        let actual = spans::from_elements(&elements);
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            c["spans"],
            "{i} spans"
        );
        assert_eq!(
            serde_json::to_value(spans::strip(actual.clone())).unwrap(),
            c["stripped"],
            "{i} strip"
        );
        assert_eq!(
            serde_json::to_value(spans::slice(
                &actual,
                c["slice"][0].as_u64().unwrap() as usize,
                c["slice"][1].as_u64().unwrap() as usize
            ))
            .unwrap(),
            c["sliced"],
            "{i} slice"
        );
        let tokens = c["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                if t[0] == "text" {
                    Token::Text(t[1].as_str().unwrap().into())
                } else {
                    Token::Formula(Kind::Omml, t[1].as_str().unwrap().into())
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(
                spans::with_formulas(&elements, c["raw"].as_str().unwrap(), &tokens).0
            )
            .unwrap(),
            c["formula"],
            "{i} formulas"
        );
    }
}
