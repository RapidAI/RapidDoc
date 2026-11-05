//! Original PPTX retry normalizer, expressed as shared package overlays.
use super::geometry::{child, P};
use crate::{
    package::{self, Package},
    xml, Result,
};
use std::{collections::BTreeSet, ops::Range};
const MC: &str = "http://schemas.openxmlformats.org/markup-compatibility/2006";
fn expected(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "officeDocument" => {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
        }
        "slide" => "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        "slideMaster" => {
            "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"
        }
        "slideLayout" => {
            "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"
        }
        "notesSlide" => {
            "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"
        }
        "notesMaster" => {
            "application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"
        }
        "theme" => "application/vnd.openxmlformats-officedocument.theme+xml",
        _ => return None,
    })
}
fn translate(mut text: String) -> String {
    // Keep the source's ordered byte substitutions, especially longer
    // relationship URIs before their common prefix.
    for (tail, target) in [
        (
            "officeDocument/relationships/metadata/thumbnail",
            "package/2006/relationships/metadata/thumbnail",
        ),
        (
            "officeDocument/relationships/customProperties",
            "officeDocument/2006/relationships/custom-properties",
        ),
        (
            "officeDocument/relationships/extendedProperties",
            "officeDocument/2006/relationships/extended-properties",
        ),
        (
            "officeDocument/relationships",
            "officeDocument/2006/relationships",
        ),
        ("drawingml/main", "drawingml/2006/main"),
        ("officeDocument/math", "officeDocument/2006/math"),
        (
            "officeDocument/customProperties",
            "officeDocument/2006/custom-properties",
        ),
        (
            "officeDocument/extendedProperties",
            "officeDocument/2006/extended-properties",
        ),
        (
            "officeDocument/docPropsVTypes",
            "officeDocument/2006/docPropsVTypes",
        ),
        ("officeDocument/oleObject", "officeDocument/2006/oleObject"),
        ("drawingml/chart", "drawingml/2006/chart"),
        ("presentationml/main", "presentationml/2006/main"),
    ] {
        text = text.replace(
            &format!("http://purl.oclc.org/ooxml/{tail}"),
            &format!("http://schemas.openxmlformats.org/{target}"),
        );
    }
    text
}
fn missing_namespace(mut text: String) -> String {
    if !text.contains("w:") || text.contains("xmlns:w=") {
        return text;
    }
    static ROOT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = ROOT.get_or_init(|| {
        regex::Regex::new(r"<(?:[A-Za-z_][A-Za-z0-9_.-]*:)?[A-Za-z_][A-Za-z0-9_.-]*(?:\s|/?>)")
            .unwrap()
    });
    let Some(root) = re.find(&text) else {
        return text;
    };
    let Some(end) = text[root.start()..].find('>').map(|n| n + root.start()) else {
        return text;
    };
    let insert = if end > 0 && text.as_bytes()[end - 1] == b'/' {
        end - 1
    } else {
        end
    };
    text.insert_str(
        insert,
        " xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"",
    );
    text
}
fn lift(text: &str, n: roxmltree::Node<'_, '_>) -> String {
    let raw = &text[n.range()];
    let Some(end) = raw.find('>') else {
        return raw.into();
    };
    let start = &raw[..end];
    let mut attrs = String::new();
    for ns in n.namespaces() {
        if ns.name() == Some("xml") {
            continue;
        }
        let name = ns
            .name()
            .map(|p| format!("xmlns:{p}"))
            .unwrap_or_else(|| "xmlns".into());
        let pattern = format!(r#"(?:^|\s){}\s*="#, regex::escape(&name));
        if !regex::Regex::new(&pattern).unwrap().is_match(start) {
            attrs.push_str(&format!(" {name}=\"{}\"", crate::chart::escape(ns.uri())));
        }
    }
    let insert = if start.ends_with('/') { end - 1 } else { end };
    format!("{}{}{}", &raw[..insert], attrs, &raw[insert..])
}
fn content_parts(text: String) -> String {
    if !text.contains("AlternateContent") || !text.contains("contentPart") {
        return text;
    }
    let Ok(doc) = xml::parse(&text) else {
        return text;
    };
    let mut edits: Vec<(Range<usize>, String)> = vec![];
    for n in doc
        .root_element()
        .descendants()
        .skip(1)
        .filter(|n| n.has_tag_name((MC, "AlternateContent")))
    {
        if edits
            .iter()
            .any(|(r, _)| r.start <= n.range().start && r.end >= n.range().end)
        {
            continue;
        }
        let (Some(choice), Some(fallback)) = (child(n, MC, "Choice"), child(n, MC, "Fallback"))
        else {
            continue;
        };
        if !choice
            .children()
            .any(|n| n.has_tag_name((P, "contentPart")))
        {
            continue;
        }
        let shapes = fallback
            .children()
            .filter(|n| {
                n.tag_name().namespace() == Some(P)
                    && matches!(
                        n.tag_name().name(),
                        "sp" | "grpSp" | "graphicFrame" | "cxnSp" | "pic"
                    )
            })
            .collect::<Vec<_>>();
        if shapes.is_empty() {
            continue;
        }
        edits.push((
            n.range(),
            shapes.into_iter().map(|n| lift(&text, n)).collect(),
        ));
    }
    drop(doc);
    let mut out = text;
    for (range, value) in edits.into_iter().rev() {
        out.replace_range(range, &value);
    }
    out
}
pub fn apply(package: &mut Package) -> Result<Vec<String>> {
    let mut changed = vec![];
    let mut skipped = BTreeSet::new();
    for name in package.names().to_vec() {
        if let Err(e) = package.validate_member(&name) {
            if !name.starts_with("ppt/media/") {
                return Err(e);
            }
            package.omit(&name);
            skipped.insert(name.clone());
            changed.push(name);
        }
    }
    for name in package
        .names()
        .to_vec()
        .into_iter()
        .filter(|n| n.ends_with(".xml") || n.ends_with(".rels"))
    {
        let bytes = package.required(&name)?;
        // The original replacements operate on bytes. In particular they do
        // not silently repair Strict namespace strings inside UTF-16 XML.
        let Ok(raw) = std::str::from_utf8(&bytes) else {
            continue;
        };
        let mut text = translate(raw.into());
        if name.ends_with(".rels") && !skipped.is_empty() {
            if let Ok(doc) = xml::parse(&text) {
                let removals = doc
                    .root_element()
                    .children()
                    .filter(|n| {
                        n.is_element()
                            && n.tag_name().name() == "Relationship"
                            && n.attribute("TargetMode") != Some("External")
                    })
                    .filter(|n| {
                        package::resolve_relationship_target(&name, n.attribute("Target"))
                            .is_some_and(|p| skipped.contains(&p))
                    })
                    .map(|n| n.range())
                    .collect::<Vec<_>>();
                drop(doc);
                for r in removals.into_iter().rev() {
                    text.replace_range(r, "");
                }
            }
        }
        if name.ends_with(".xml") {
            text = content_parts(missing_namespace(text));
        }
        if text.as_bytes() != bytes.as_ref() {
            package.patch(name.clone(), text.into_bytes().into());
            changed.push(name);
        }
    }
    if crate::docx::normalize::repair_types_for(package, expected)?.is_some() {
        changed.push(package::CONTENT_TYPES.into());
    }
    changed.sort();
    changed.dedup();
    Ok(changed)
}
