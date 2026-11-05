//! Original PdfModel final visible text normalization, preserving formulas, spans and HTML payloads.
use crate::{
    legacy_page::{invalid, limit, validate_tree, Work},
    legacy_table_merge::html::{Dom, Node},
};
use rapiddoc_core::Error;
use serde_json::Value;
use std::sync::OnceLock;
fn config() -> &'static Value {
    static C: OnceLock<Value> = OnceLock::new();
    C.get_or_init(|| serde_json::from_str(include_str!("native_model_normalization.json")).unwrap())
}
fn mapped(c: char) -> char {
    if matches!(c as u32,0xff21..=0xff3a|0xff41..=0xff5a|0xff10..=0xff19)
        || "：．／＼－＿％＋＝＠＃＆＊".contains(c)
    {
        char::from_u32(c as u32 - 0xfee0).unwrap()
    } else {
        c
    }
}
fn normalize(s: &str, w: &mut Work) -> Result<String, Error> {
    w.charge(s.len())?;
    let cs = s.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '\\' && cs.get(i + 1).is_some_and(|c| matches!(c, '(' | '[')) {
            let close = if cs[i + 1] == '(' { ')' } else { ']' };
            let start = i;
            i += 2;
            while i + 1 < cs.len() && !(cs[i] == '\\' && cs[i + 1] == close) {
                w.charge(1)?;
                i += 1
            }
            i = if i + 1 < cs.len() { i + 2 } else { cs.len() };
            out.extend(&cs[start..i]);
        } else {
            out.push(mapped(cs[i]));
            i += 1
        }
    }
    Ok(out)
}
fn parts(parts: &[String], w: &mut Work) -> Result<Vec<String>, Error> {
    let s = parts.join("");
    let n = normalize(&s, w)?.chars().collect::<Vec<_>>();
    let mut at = 0;
    let mut out = vec![];
    for p in parts {
        let count = p.chars().count();
        out.push(n[at..at + count].iter().collect());
        at += count;
    }
    Ok(out)
}
fn gather(
    spans: &Value,
    path: &mut Vec<usize>,
    out: &mut Vec<(Option<Vec<usize>>, String)>,
    depth: usize,
) -> Result<(), Error> {
    if depth > rapiddoc_core::resources::limit(64) {
        return Err(limit());
    }
    for (i, s) in spans.as_array().ok_or_else(invalid)?.iter().enumerate() {
        path.push(i);
        if s.is_object() && s["type"] == "text" && s["content"].is_string() {
            out.push((Some(path.clone()), s["content"].as_str().unwrap().into()))
        } else if s.is_object() && s["type"] == "hyperlink" && s["content"].is_array() {
            gather(&s["content"], path, out, depth + 1)?
        } else {
            out.push((None, "\0".into()))
        }
        path.pop();
    }
    Ok(())
}
fn spans(v: &mut Value, w: &mut Work) -> Result<(), Error> {
    let mut entries = vec![];
    gather(v, &mut vec![], &mut entries, 0)?;
    let normalized = parts(
        &entries.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>(),
        w,
    )?;
    for ((path, original), text) in entries.into_iter().zip(normalized) {
        if text == original {
            continue;
        }
        if let Some(path) = path {
            let mut s = &mut *v;
            for (i, &index) in path.iter().enumerate() {
                s = &mut s[index];
                if i + 1 < path.len() {
                    s = &mut s["content"];
                }
            }
            s["content"] = Value::String(text);
        }
    }
    Ok(())
}
fn opaque(n: &Node) -> bool {
    config()["opaque"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t.as_str() == n.tag.rsplit(':').next())
        || n.attrs.get("data-block-type").is_some_and(|t| {
            [
                "equation",
                "code",
                "code_body",
                "algorithm",
                "algorithm_body",
            ]
            .contains(&t.as_str())
        })
        || n.attrs.contains_key("data-docvortex-latex")
        || n.attrs.contains_key("data-formula-display")
        || n.attrs
            .get("class")
            .is_some_and(|s| s.split_whitespace().any(|s| s == "docvortex-math"))
}
fn cell_parts(dom: &Dom, id: usize, out: &mut Vec<(Option<usize>, String)>) {
    let n = &dom.nodes[id];
    if n.tag.is_empty() {
        out.push((Some(id), n.text.clone()));
    } else if !n.tag.starts_with('!') {
        if opaque(n) || ["table", "br", "hr", "img"].contains(&n.tag.as_str()) {
            out.push((None, "\0".into()));
        } else {
            for &child in &n.children {
                cell_parts(dom, child, out)
            }
        }
    }
}
fn table(s: &str, w: &mut Work) -> Result<String, Error> {
    if !s.chars().any(|c| mapped(c) != c) && !s.contains("&#") {
        return Ok(s.into());
    }
    let mut dom = Dom::parse_normalization(s, w)?;
    let cells = dom.find(0, &["td", "th"]);
    let mut changed = false;
    for cell in cells {
        let mut id = Some(cell);
        let mut protected = false;
        while let Some(i) = id {
            if opaque(&dom.nodes[i]) {
                protected = true;
                break;
            }
            id = dom.nodes[i].parent;
        }
        if protected {
            continue;
        }
        let mut entries = vec![];
        for &child in &dom.nodes[cell].children {
            cell_parts(&dom, child, &mut entries)
        }
        let n = parts(
            &entries.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>(),
            w,
        )?;
        for ((id, original), replacement) in entries.into_iter().zip(n) {
            if let Some(id) = id {
                if replacement != original {
                    dom.nodes[id].text = replacement;
                    changed = true;
                }
            }
        }
    }
    Ok(if changed {
        dom.normalized_html()
    } else {
        s.into()
    })
}
pub fn normalize_pages(pages: &Value) -> Result<Value, Error> {
    validate_tree(pages)?;
    let mut out = pages.clone();
    normalize_with_work(&mut out, &mut Work(0))?;
    Ok(out)
}
pub(super) fn normalize_with_work(pages: &mut Value, w: &mut Work) -> Result<(), Error> {
    let pages = pages.as_array_mut().ok_or_else(invalid)?;
    if pages.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    for p in pages {
        for b in p.as_array_mut().ok_or_else(invalid)? {
            w.charge(1)?;
            let ty = b["type"].clone();
            if config()["natural"].as_array().unwrap().contains(&ty) {
                if let Some(s) = b["content"].as_str() {
                    b["content"] = Value::String(normalize(s, w)?)
                } else if b["content"].is_array() {
                    spans(&mut b["content"], w)?
                }
            } else if ty == "table" || ty == "table_body" {
                if let Some(s) = b["content"].as_str() {
                    b["content"] = Value::String(table(s, w)?);
                }
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_model_normalization() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-model-normalization.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got =
                normalize_pages(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
            assert_eq!(
                normalize_pages(&got).unwrap(),
                got,
                "{} idempotence",
                c["label"]
            );
        }
    }
}
