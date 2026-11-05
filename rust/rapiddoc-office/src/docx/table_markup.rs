//! Mammoth HTML empty stripping/collapse and the local colspan correction.
use super::integer_text;
use num_bigint::BigInt;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Html {
    Text(String),
    Element {
        names: Vec<String>,
        attributes: BTreeMap<String, String>,
        collapsible: bool,
        separator: Option<String>,
        children: Vec<Html>,
    },
    NoteReference {
        kind: String,
        id: String,
    },
    ResourceFragment(String),
    ForceWrite,
}
impl Html {
    pub fn element(name: &str, collapsible: bool, children: Vec<Self>) -> Self {
        Self::Element {
            names: vec![name.into()],
            attributes: BTreeMap::new(),
            collapsible,
            separator: None,
            children,
        }
    }
    pub fn attribute(mut self, key: &str, value: impl Into<String>) -> Self {
        if let Self::Element { attributes, .. } = &mut self {
            attributes.insert(key.into(), value.into());
        }
        self
    }
    pub(super) fn alternatives(mut self, names: &[&str]) -> Self {
        if let Self::Element { names: target, .. } = &mut self {
            *target = names.iter().map(|s| (*s).into()).collect();
        }
        self
    }
}
fn void(name: &str) -> bool {
    matches!(name, "br" | "hr" | "img" | "input")
}
pub(super) fn strip_empty(nodes: Vec<Html>) -> Vec<Html> {
    nodes
        .into_iter()
        .filter_map(|node| match node {
            Html::Text(ref t) if t.is_empty() => None,
            Html::Element {
                names,
                attributes,
                collapsible,
                separator,
                children,
            } => {
                let children = strip_empty(children);
                if children.is_empty() && !void(&names[0]) {
                    None
                } else {
                    Some(Html::Element {
                        names,
                        attributes,
                        collapsible,
                        separator,
                        children,
                    })
                }
            }
            node => Some(node),
        })
        .collect()
}
fn collapsing_add(output: &mut Vec<Html>, node: Html) {
    match (output.last_mut(), node) {
        (
            Some(Html::Element {
                names: last,
                attributes: left,
                children: previous,
                ..
            }),
            Html::Element {
                names,
                attributes,
                collapsible: true,
                separator,
                children,
            },
        ) if names.contains(&last[0]) && *left == attributes => {
            if let Some(s) = separator.filter(|s| !s.is_empty()) {
                previous.push(Html::Text(s));
            }
            for child in children {
                collapsing_add(previous, child);
            }
        }
        (_, node) => output.push(node),
    }
}
pub(super) fn collapse(nodes: Vec<Html>) -> Vec<Html> {
    let mut output = Vec::new();
    for mut node in nodes {
        if let Html::Element { children, .. } = &mut node {
            *children = collapse(std::mem::take(children));
        }
        collapsing_add(&mut output, node);
    }
    output
}
fn escape(text: &str, output: &mut String) {
    // Mammoth HtmlWriter escapes double quotes in text as well as attributes.
    for c in text.chars() {
        match c {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            _ => output.push(c),
        }
    }
}
pub(super) fn write(nodes: &[Html], output: &mut String) {
    write_with_void_suffix(nodes, output, " />");
}
/// BeautifulSoup's table boundary uses compact XML void tags; Mammoth's raw
/// serializer above keeps its own space before the slash.
pub(super) fn write_table(nodes: &[Html], output: &mut String) {
    write_with_void_suffix(nodes, output, "/>");
}
fn write_with_void_suffix(nodes: &[Html], output: &mut String, suffix: &str) {
    for node in nodes {
        match node {
            Html::Text(t) => escape(t, output),
            Html::ForceWrite => (),
            Html::NoteReference { .. } => {
                unreachable!("note reference must be resolved before serialization")
            }
            Html::ResourceFragment(_) => {
                unreachable!("resource must be resolved before serialization")
            }
            Html::Element {
                names,
                attributes,
                children,
                ..
            } => {
                let name = &names[0];
                output.push('<');
                output.push_str(name);
                for (key, value) in attributes {
                    output.push(' ');
                    output.push_str(key);
                    output.push_str("=\"");
                    escape(value, output);
                    output.push('"');
                }
                if void(name) && children.is_empty() {
                    output.push_str(suffix);
                } else {
                    output.push('>');
                    write_with_void_suffix(children, output, suffix);
                    output.push_str("</");
                    output.push_str(name);
                    output.push('>');
                }
            }
        }
    }
}
pub fn serialize(nodes: Vec<Html>) -> String {
    let mut output = String::new();
    write(&collapse(strip_empty(nodes)), &mut output);
    output
}

/// Port of _normalize_table_colspans operating on the existing HTML tree, not
/// an HTML serialize/BeautifulSoup/serialize roundtrip. Traversal deliberately
/// includes nested rows/cells, and rowspan protects the complete outer table.
/// Return false without mutations on invalid integer attributes, matching the
/// original function's return of its unchanged input on exceptions.
pub fn normalize_colspans(nodes: &mut [Html]) -> bool {
    fn validate(nodes: &[Html]) -> bool {
        nodes.iter().all(|n| match n {
            Html::Element {
                names, children, ..
            } => {
                if names[0] == "table" {
                    let mut guarded = false;
                    for path in matching_paths(children, &["td", "th"]) {
                        let Html::Element { attributes, .. } = node_at(children, &path) else {
                            unreachable!()
                        };
                        let Some(value) = span(attributes.get("rowspan")) else {
                            return false;
                        };
                        if value > 1.into() {
                            guarded = true;
                            break;
                        }
                    }
                    if !guarded {
                        for row in matching_paths(children, &["tr"]) {
                            let cells = child_nodes(node_at(children, &row));
                            for path in matching_paths(cells, &["td", "th"]) {
                                let Html::Element { attributes, .. } = node_at(cells, &path) else {
                                    unreachable!()
                                };
                                if span(attributes.get("colspan")).is_none() {
                                    return false;
                                }
                            }
                        }
                    }
                }
                validate(children)
            }
            _ => true,
        })
    }
    fn normalize(nodes: &mut [Html]) -> bool {
        let mut changed = false;
        for node in nodes {
            if let Html::Element {
                names, children, ..
            } = node
            {
                if names[0] == "table" {
                    changed |= normalize_one(children);
                }
                changed |= normalize(children);
            }
        }
        changed
    }
    if validate(nodes) {
        normalize(nodes)
    } else {
        false
    }
}
fn span(value: Option<&String>) -> Option<BigInt> {
    match value {
        None => Some(1.into()),
        Some(s) => integer_text(Some(s))?.parse().ok(),
    }
}
pub(super) fn child_nodes(node: &Html) -> &[Html] {
    if let Html::Element { children, .. } = node {
        children
    } else {
        &[]
    }
}
pub(super) fn matching_paths(nodes: &[Html], names: &[&str]) -> Vec<Vec<usize>> {
    fn collect(
        nodes: &[Html],
        names: &[&str],
        path: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        for (index, node) in nodes.iter().enumerate() {
            path.push(index);
            if let Html::Element {
                names: tag,
                children,
                ..
            } = node
            {
                if names.contains(&tag[0].as_str()) {
                    output.push(path.clone());
                }
                collect(children, names, path, output);
            }
            path.pop();
        }
    }
    let mut output = Vec::new();
    collect(nodes, names, &mut Vec::new(), &mut output);
    output
}
pub(super) fn node_at<'a>(mut nodes: &'a [Html], path: &[usize]) -> &'a Html {
    for (index, part) in path.iter().enumerate() {
        let node = &nodes[*part];
        if index + 1 == path.len() {
            return node;
        }
        nodes = child_nodes(node);
    }
    unreachable!("non-empty node path")
}
pub(super) fn node_at_mut<'a>(nodes: &'a mut [Html], path: &[usize]) -> &'a mut Html {
    let node = &mut nodes[path[0]];
    if path.len() == 1 {
        return node;
    }
    if let Html::Element { children, .. } = node {
        node_at_mut(children, &path[1..])
    } else {
        unreachable!("element parent")
    }
}
fn attribute_span(node: &Html, key: &str) -> BigInt {
    if let Html::Element { attributes, .. } = node {
        span(attributes.get(key)).expect("validated HTML integer")
    } else {
        unreachable!("cell element")
    }
}
fn normalize_one(nodes: &mut [Html]) -> bool {
    let cells = matching_paths(nodes, &["td", "th"]);
    if cells
        .iter()
        .any(|path| attribute_span(node_at(nodes, path), "rowspan") > 1.into())
    {
        return false;
    }
    let rows = matching_paths(nodes, &["tr"]);
    let mut counts = Vec::new();
    let mut frequencies = BTreeMap::<BigInt, (usize, usize)>::new();
    for (index, path) in rows.iter().enumerate() {
        let contents = child_nodes(node_at(nodes, path));
        let count: BigInt = matching_paths(contents, &["td", "th"])
            .iter()
            .map(|path| attribute_span(node_at(contents, path), "colspan"))
            .sum();
        frequencies.entry(count.clone()).or_insert((0, index)).0 += 1;
        counts.push(count);
    }
    if frequencies.len() <= 1 {
        return false;
    }
    let target = frequencies
        .iter()
        .max_by(|(_, a), (_, b)| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
        .unwrap()
        .0
        .clone();
    let mut changed = false;
    for (row, count) in rows.iter().zip(counts) {
        if count <= target {
            continue;
        }
        let mut excess = count - &target;
        let cells = matching_paths(child_nodes(node_at(nodes, row)), &["td", "th"]);
        for cell in cells {
            if excess <= 0.into() {
                break;
            }
            let mut full = row.clone();
            full.extend(cell);
            let node = node_at_mut(nodes, &full);
            let original = attribute_span(node, "colspan");
            if original > 1.into() {
                let reduce = (&original - BigInt::from(1)).min(excess.clone());
                let next = original - &reduce;
                if let Html::Element { attributes, .. } = node {
                    if next == 1.into() {
                        attributes.remove("colspan");
                    } else {
                        attributes.insert("colspan".into(), next.to_string());
                    }
                }
                excess -= reduce;
                changed = true;
            }
        }
    }
    changed
}
