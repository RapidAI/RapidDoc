//! DocVortex package_normalizer.py policy, applied as shared ZIP overlays.
//! Native namespace readers accept Strict directly; no rewritten ZIP copy.
use crate::{
    package::{self, Package},
    xml, Result,
};
use std::collections::BTreeSet;
#[derive(Default, Debug, serde::Serialize)]
pub struct Report {
    pub skipped_members: Vec<String>,
    pub removed_relationships: Vec<(String, String)>,
    pub repaired_content_types: Vec<(String, String)>,
}
fn target(name: &str, value: Option<&str>) -> Option<String> {
    // Preserve the native reader's existing support for a main part at ZIP
    // root; Python's historical normalizer helper lacks this relative case.
    if let Some(source) = name
        .strip_prefix("_rels/")
        .and_then(|n| n.strip_suffix(".rels"))
    {
        if !source.is_empty() && !source.contains('/') {
            return package::resolve_part_target(source, value?);
        }
    }
    package::resolve_relationship_target(name, value)
}
pub fn apply(package: &mut Package) -> Result<Report> {
    let members = package.names().iter().cloned().collect::<BTreeSet<_>>();
    let mut reachable = BTreeSet::from([package::CONTENT_TYPES.to_owned()]);
    let mut complete = members.contains(package::ROOT_RELS);
    let mut queue = if complete {
        vec![package::ROOT_RELS.to_owned()]
    } else {
        vec![]
    };
    let mut visited = BTreeSet::new();
    while let Some(name) = queue.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        reachable.insert(name.clone());
        // Compression errors in relationship traversal are fatal in Python,
        // even when an unrelated corrupt member could otherwise be skipped.
        let data = package.required(&name)?;
        let Ok(text) = xml::decode(&data) else {
            complete = false;
            continue;
        };
        let Ok(tree) = xml::parse(&text) else {
            complete = false;
            continue;
        };
        for rel in tree
            .root_element()
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
        {
            if rel.attribute("TargetMode") == Some("External") {
                continue;
            }
            if let Some(target) = target(&name, rel.attribute("Target")) {
                if members.contains(&target) {
                    reachable.insert(target.clone());
                    if let Some(rels) = package::part_relationships_member(&target)
                        .filter(|p| members.contains(p) && !visited.contains(p))
                    {
                        queue.push(rels);
                    }
                }
            }
        }
    }
    let mut report = Report::default();
    for name in package.names().to_vec() {
        if let Err(error) = package.validate_member(&name) {
            let skippable = name.starts_with("word/media/")
                || name.replace('\\', "/").starts_with("word/embeddings/")
                || (complete && !reachable.contains(&name));
            if !skippable {
                return Err(error);
            }
            package.omit(&name);
            report.skipped_members.push(name);
        }
    }
    for name in package
        .names()
        .to_vec()
        .into_iter()
        .filter(|n| n.ends_with(".rels"))
    {
        let data = package.required(&name)?;
        let Ok(text) = xml::decode(&data) else {
            continue;
        };
        let Ok(tree) = xml::parse(&text) else {
            continue;
        };
        let mut removed = vec![];
        for rel in tree
            .root_element()
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
        {
            if rel.attribute("TargetMode") == Some("External") {
                continue;
            }
            let target = target(&name, rel.attribute("Target"));
            if target.is_none_or(|p| !package.contains(&p)) {
                removed.push(rel.range());
                report
                    .removed_relationships
                    .push((name.clone(), rel.attribute("Id").unwrap_or("").into()));
            }
        }
        if !removed.is_empty() {
            let mut output = text.into_owned();
            for range in removed.into_iter().rev() {
                output.replace_range(range, "");
            }
            package.patch(name, xml::encode_utf8(output).into());
        }
    }
    report.repaired_content_types = repair_types(package)?.unwrap_or_default();
    Ok(report)
}
fn expected(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "officeDocument" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
        }
        "styles" => "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
        "numbering" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"
        }
        "footnotes" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"
        }
        "endnotes" => "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml",
        "comments" => "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
        "settings" => "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
        "webSettings" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.webSettings+xml"
        }
        "fontTable" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"
        }
        "theme" => "application/vnd.openxmlformats-officedocument.theme+xml",
        _ => return None,
    })
}
fn escape(value: &str) -> String {
    quick_xml::escape::escape(value).into_owned()
}
fn repair_types(package: &mut Package) -> Result<Option<Vec<(String, String)>>> {
    repair_types_for(package, expected)
}
pub(crate) fn repair_types_for(
    package: &mut Package,
    expected: fn(&str) -> Option<&'static str>,
) -> Result<Option<Vec<(String, String)>>> {
    let Some(bytes) = package.read(package::CONTENT_TYPES)? else {
        return Ok(None);
    };
    let Ok(text) = xml::decode(&bytes) else {
        return Ok(None);
    };
    let Ok(tree) = xml::parse(&text) else {
        return Ok(None);
    };
    const NS: &str = "http://schemas.openxmlformats.org/package/2006/content-types";
    let overrides = tree
        .root_element()
        .children()
        .filter(|n| {
            n.is_element()
                && n.tag_name().name() == "Override"
                && n.tag_name().namespace() == Some(NS)
        })
        .filter_map(|n| {
            let name = n
                .attribute("PartName")
                .unwrap_or("")
                .replace('\\', "/")
                .trim_start_matches('/')
                .to_owned();
            if name.is_empty() {
                None
            } else {
                Some((name, n))
            }
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut queue = vec![];
    let mut seen = BTreeSet::new();
    let mut pending = vec![];
    let Some(root) = package.read_xml(package::ROOT_RELS)? else {
        return Ok(None);
    };
    let Ok(rels) = xml::parse(&root) else {
        return Ok(None);
    };
    for rel in rels
        .root_element()
        .children()
        .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
    {
        if rel.attribute("Type").unwrap_or("").rsplit('/').next() == Some("officeDocument") {
            if let Some(part) =
                target(package::ROOT_RELS, rel.attribute("Target")).filter(|p| package.contains(p))
            {
                queue.push((part, expected("officeDocument").unwrap()));
            }
        }
    }
    while let Some((part, ct)) = queue.pop() {
        if !seen.insert(part.clone()) {
            continue;
        }
        if overrides
            .get(&part)
            .and_then(|n| n.attribute("ContentType"))
            != Some(ct)
        {
            pending.push((part.clone(), ct.to_owned()));
        }
        let Some(name) = package::part_relationships_member(&part) else {
            continue;
        };
        let Some(rels) = package.read_xml(&name)? else {
            continue;
        };
        let Ok(rels) = xml::parse(&rels) else {
            return Ok(None);
        };
        for rel in rels
            .root_element()
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
        {
            if let Some(ct) = expected(
                rel.attribute("Type")
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or(""),
            ) {
                if let Some(part) =
                    target(&name, rel.attribute("Target")).filter(|p| package.contains(p))
                {
                    queue.push((part, ct));
                }
            }
        }
    }
    if pending.is_empty() {
        return Ok(None);
    }
    let mut edits = vec![];
    let mut added = String::new();
    for (part, ct) in &pending {
        if let Some(node) = overrides.get(part) {
            if let Some(attribute) = node.attribute_node("ContentType") {
                edits.push((attribute.range_value(), escape(ct)));
            } else {
                let start = node.range().start + 1;
                let end = start
                    + text[start..]
                        .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
                        .unwrap();
                edits.push((end..end, format!(" ContentType=\"{}\"", escape(ct))));
            }
        } else {
            added.push_str(&format!(
                "<Override xmlns=\"{NS}\" PartName=\"/{}\" ContentType=\"{}\"/>",
                escape(part),
                escape(ct)
            ));
        }
    }
    if !added.is_empty() {
        let root = tree.root_element();
        let range = root.range();
        let raw = &text[range.clone()];
        if raw.ends_with("/>") {
            let tag = raw[1..]
                .split(|c: char| c.is_whitespace() || c == '/' || c == '>')
                .next()
                .unwrap();
            edits.push((range.end - 2..range.end, format!(">{added}</{tag}>")));
        } else {
            let pos = range.start + raw.rfind("</").unwrap();
            edits.push((pos..pos, added));
        }
    }
    edits.sort_by_key(|(range, _)| std::cmp::Reverse(range.start));
    let mut output = text.into_owned();
    for (range, value) in edits {
        output.replace_range(range, &value);
    }
    package.patch(
        package::CONTENT_TYPES.into(),
        xml::encode_utf8(output).into(),
    );
    Ok(Some(pending))
}
