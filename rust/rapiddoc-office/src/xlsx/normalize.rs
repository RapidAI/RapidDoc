//! Original XLSX package_normalizer policy as ZIP overlays, on failed input only.
use super::*;
use std::{ops::Range, sync::OnceLock};
fn worksheet(name: &str) -> bool {
    name.starts_with("xl/worksheets/")
        && name.ends_with(".xml")
        && !name.starts_with("xl/worksheets/_rels/")
}
pub fn apply(package: &mut Package) -> Result<Vec<String>> {
    let mut changed = vec![];
    for name in package.names().to_vec() {
        // Python's retry reads every ZIP member, including CRC validation.
        if name != "xl/sharedStrings.xml" && name != "xl/styles.xml" && !worksheet(&name) {
            package.validate_member(&name)?;
            continue;
        }
        let Some(bytes) = package.read(&name)? else {
            continue;
        };
        let Some(normalized) = member(&name, &bytes)? else {
            continue;
        };
        package.patch(name.clone(), normalized.into());
        changed.push(name);
    }
    Ok(changed)
}
fn member(name: &str, bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    let Ok(text) = xml::decode(bytes) else {
        return Ok(None);
    };
    let Ok(tree) = xml::parse(&text) else {
        return Ok(None);
    };
    let root = tree.root_element();
    let mut edits: Vec<(Range<usize>, String)> = vec![];
    if name == "xl/sharedStrings.xml" {
        for underline in root.descendants().skip(1).filter(|n| is(*n, "u")) {
            for ns in [None, Some(NS)] {
                if let Some(a) = underline
                    .attributes()
                    .find(|a| a.name() == "val" && a.namespace() == ns)
                {
                    if a.value().trim_matches(crate::docx::space).is_empty() {
                        edits.push((a.range(), String::new()));
                        break;
                    }
                    if !matches!(
                        a.value(),
                        "single" | "double" | "singleAccounting" | "doubleAccounting" | "none"
                    ) {
                        break;
                    }
                }
            }
        }
    } else if name == "xl/styles.xml" {
        if let Some(fills) = child(root, "fills") {
            for fill in fills
                .children()
                .filter(|n| is(*n, "fill") && !n.children().any(|c| c.is_element()))
            {
                let range = fill.range();
                let raw = &text[range.clone()];
                let inner = format!("<patternFill xmlns=\"{NS}\"/>");
                if raw.ends_with("/>") {
                    let tag = raw[1..]
                        .split(|c: char| c.is_whitespace() || c == '/' || c == '>')
                        .next()
                        .unwrap();
                    edits.push((range.end - 2..range.end, format!(">{inner}</{tag}>")));
                } else {
                    let end = range.start + raw.rfind("</").unwrap();
                    edits.push((end..end, inner));
                }
            }
        }
    } else if worksheet(name) {
        for filter in root.descendants().skip(1).filter(|n| is(*n, "autoFilter")) {
            let Some(attr) = filter.attribute_node("ref") else {
                continue;
            };
            static ROW: OnceLock<regex::Regex> = OnceLock::new();
            let re = ROW
                .get_or_init(|| regex::Regex::new(r"^\$?([1-9][0-9]*):\$?([1-9][0-9]*)$").unwrap());
            let Some(m) = re.captures(attr.value()) else {
                continue;
            };
            let a = m[1].parse::<num_bigint::BigInt>().unwrap();
            let b = m[2].parse::<num_bigint::BigInt>().unwrap();
            if a > b {
                continue;
            }
            let (c, d) = columns(root);
            edits.push((attr.range_value(), format!("{c}{a}:{d}{b}")));
        }
    }
    if edits.is_empty() {
        return Ok(None);
    }
    edits.sort_by_key(|(r, _)| std::cmp::Reverse(r.start));
    let mut text = text.into_owned();
    for (range, value) in edits {
        text.replace_range(range, &value);
    }
    Ok(Some(xml::encode_utf8(text)))
}
fn column_bounds(raw: &str) -> Option<(String, String)> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(
            r"^\$?([A-Za-z]{1,3})(?:\$?[1-9][0-9]*)?(?::\$?([A-Za-z]{1,3})(?:\$?[1-9][0-9]*)?)?$",
        )
        .unwrap()
    });
    let s = raw.trim_matches(crate::docx::space);
    let m = re.captures(s)?;
    // Bare columns and mixed cell/column bounds are not original dimension forms.
    let has_digit = s.bytes().any(|b| b.is_ascii_digit());
    if !has_digit && !s.contains(':') {
        return None;
    }
    if s.contains(':') {
        let (a, b) = s.split_once(':')?;
        if a.bytes().any(|b| b.is_ascii_digit()) != b.bytes().any(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    let a = m[1].to_uppercase();
    let b = m
        .get(2)
        .map(|m| m.as_str().to_uppercase())
        .unwrap_or_else(|| a.clone());
    Some((a, b))
}
fn columns(root: Node<'_, '_>) -> (String, String) {
    if let Some(v) = child(root, "dimension")
        .and_then(|n| n.attribute("ref"))
        .and_then(column_bounds)
    {
        return v;
    }
    let mut indices = vec![];
    if let Some(data) = child(root, "sheetData") {
        for c in data.descendants().skip(1).filter(|n| is(*n, "c")) {
            let Some(s) = c.attribute("r") else { continue };
            if s.contains(':') {
                continue;
            }
            if let Some((name, _)) =
                column_bounds(s).filter(|_| s.bytes().any(|b| b.is_ascii_digit()))
            {
                indices.push(
                    name.bytes()
                        .fold(0u32, |n, b| n * 26 + u32::from(b - b'A' + 1)),
                );
            }
        }
    }
    fn name(mut n: u32) -> String {
        let mut b = vec![];
        while n > 0 {
            n -= 1;
            b.push(b'A' + (n % 26) as u8);
            n /= 26
        }
        b.reverse();
        String::from_utf8(b).unwrap()
    }
    match (indices.iter().min(), indices.iter().max()) {
        (Some(a), Some(b)) => (name(*a), name(*b)),
        _ => ("A".into(), "XFD".into()),
    }
}
pub(super) fn validate_filter(root: Node<'_, '_>) -> Result<()> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(
            r"^\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d*)?$|^[A-Za-z]{1,3}:[A-Za-z]{1,3}$",
        )
        .unwrap()
    });
    for n in root.children().filter(|n| is(*n, "autoFilter")) {
        if n.attribute("ref").is_some_and(|s| !re.is_match(s)) {
            return Err(Error::malformed("invalid worksheet autoFilter range"));
        }
    }
    Ok(())
}
